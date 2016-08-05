import backend
import ufl
from solving import solve, annotate as solving_annotate, do_checkpoint, register_initial_conditions
import libadjoint
import assignment
import adjlinalg
import adjglobals
import utils
import misc
import compatibility

dolfin_assign = backend.Function.assign
dolfin_split  = backend.Function.split
dolfin_str    = backend.Function.__str__
dolfin_interpolate = backend.Function.interpolate
if backend.__name__ != 'dolfin':
    firedrake_project = backend.Function.project

def dolfin_adjoint_assign(self, other, annotate=None, *args, **kwargs):
    '''We also need to monkeypatch the Function.assign method, as it is often used inside
    the main time loop, and not annotating it means you get the adjoint wrong for totally
    nonobvious reasons. If anyone objects to me monkeypatching your objects, my apologies
    in advance.'''

    if self is other:
        return

    to_annotate = utils.to_annotate(annotate)
    # if we shouldn't annotate, just assign
    if not to_annotate:
        return dolfin_assign(self, other, *args, **kwargs)

    if isinstance(other, ufl.algebra.Sum) or isinstance(other, ufl.algebra.Product):
        if backend.__name__ != 'dolfin':
            errmsg = '''Cannot use Function.assign(linear combination of other Functions) yet.'''
            raise libadjoint.exceptions.LibadjointErrorNotImplemented(errmsg)
        else:
            lincom = _check_and_contract_linear_comb(other, self)
    else:
        lincom = [(other, 1.0)]

    # ignore anything not a backend.Function, unless the user insists
    if not isinstance(other, backend.Function) and (annotate is not True):
        return dolfin_assign(self, other, *args, **kwargs)

    # ignore anything that is an interpolation, rather than a straight assignment
    if hasattr(self, "function_space") and hasattr(other, "function_space"):
        if str(self.function_space()) != str(other.function_space()):
            return dolfin_assign(self, other, *args, **kwargs)


    functions, weights = zip(*lincom)

    self_var = adjglobals.adj_variables[self]
    function_vars = [adjglobals.adj_variables[function] for function in functions]

    # ignore any functions we haven't seen before -- we DON'T want to
    # annotate the assignment of initial conditions here. That happens
    # in the main solve wrapper.
    for function_var in function_vars:
        if not adjglobals.adjointer.variable_known(function_var) and not adjglobals.adjointer.variable_known(self_var) and (annotate is not True):
            [adjglobals.adj_variables.forget(function) for function in functions]
            adjglobals.adj_variables.forget(self)

            return dolfin_assign(self, other, *args, **kwargs)

    # OK, so we have a variable we've seen before. Beautiful.
    if not adjglobals.adjointer.variable_known(self_var):
        adjglobals.adj_variables.forget(self)

    out = dolfin_assign(self, other, *args, **kwargs)

    fn_space = self.function_space()
    identity_block = utils.get_identity_block(fn_space)
    dep = adjglobals.adj_variables.next(self)

    if backend.parameters["adjoint"]["record_all"]:
        adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(self)))

    rhs = LinComRHS(functions, weights, fn_space)
    register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)
    initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
    cs = adjglobals.adjointer.register_equation(initial_eq)

    do_checkpoint(cs, dep, rhs)

    return out

def dolfin_adjoint_split(self, *args, **kwargs):
    out = dolfin_split(self, *args, **kwargs)
    for i, fn in enumerate(out):
        fn.split_da = True
        fn.split_fn = self
        fn.split_i  = i
        fn.split_args = args
        fn.split_kwargs = kwargs

    return out

def dolfin_adjoint_str(self):
    if hasattr(self, "adj_name"):
        return self.adj_name
    else:
        return dolfin_str(self)

def dolfin_adjoint_interpolate(self, other, annotate=None):
    out = dolfin_interpolate(self, other)
    if annotate is True:
        assignment.register_assign(self, other, op=backend.interpolate)
        adjglobals.adjointer.record_variable(adjglobals.adj_variables[self], libadjoint.MemoryStorage(adjlinalg.Vector(self)))

    return out

if hasattr(backend.Function, 'sub'):
    dolfin_adjoint_sub = compatibility.dolfin_adjoint_sub

class Function(backend.Function):
    '''The Function class is overloaded so that you can give :py:class:`Functions` *names*. For example,

      .. code-block:: python

        u = Function(V, name="Velocity")

      This allows you to refer to the :py:class:`Function` by name throughout dolfin-adjoint, rather than
      needing to have the specific :py:class:`Function` instance available.

      For more details, see :doc:`the dolfin-adjoint documentation </documentation/misc>`.'''

    def __init__(self, *args, **kwargs):

        annotate = kwargs.pop("annotate", None)
        to_annotate = utils.to_annotate(annotate)

        if "name" in kwargs:
            self.adj_name = kwargs["name"]

            #if self.adj_name in adjglobals.function_names and to_annotate:
            #  backend.info_red("Warning: got duplicate function name %s" % self.adj_name)

            adjglobals.function_names.add(self.adj_name)
            del kwargs["name"]

        with misc.annotations(False):
            backend.Function.__init__(self, *args, **kwargs)

        if hasattr(self, 'adj_name'):
            self.rename(self.adj_name, "a Function from dolfin-adjoint")

        if to_annotate:
            if not isinstance(args[0], compatibility.function_space_type):
                if isinstance(args[0], backend.Function):
                    known = adjglobals.adjointer.variable_known(adjglobals.adj_variables[args[0]])
                else:
                    known = True

                if known or (annotate is True):
                    assignment.register_assign(self, args[0])
                else:
                    adjglobals.adj_variables.forget(args[0])

    def project(self, other, annotate=None, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Firedrake project call.'''

        to_annotate = utils.to_annotate(annotate)

        if not to_annotate:
            flag = misc.pause_annotation()

        res = firedrake_project(self, other, *args, **kwargs)

        if not to_annotate:
            misc.continue_annotation(flag)

        return res

    def assign(self, other, annotate=None, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin assign call.'''

        return dolfin_adjoint_assign(self, other, annotate=annotate, *args, **kwargs)

    def split(self, *args, **kwargs):
        return dolfin_adjoint_split(self, *args, **kwargs)

    def __str__(self):
        return dolfin_adjoint_str(self)

    def interpolate(self, other, annotate=None):
        if annotate is True and backend.parameters["adjoint"]["stop_annotating"]:
            raise AssertionError("The user insisted on annotation, but stop_annotating is True.")

        return dolfin_adjoint_interpolate(self, other, annotate)

    def copy(self, deepcopy, name, annotate=None):
        out = backend.Function.copy(self, deepcopy)
        out.rename(name, name)
        dolfin_adjoint_assign(self, out, annotate=annotate)
        return out

    if hasattr(backend.Function, 'sub'):
        def sub(self, idx, deepcopy=False):
            return dolfin_adjoint_sub(self, idx, deepcopy=deepcopy)

backend.Function.assign = dolfin_adjoint_assign # so that Functions produced inside Expression etc. get it too
if backend.__name__ == "dolfin":
    backend.Function.split  = dolfin_adjoint_split
else:
    backend.Function.project = firedrake_project
backend.Function.__str__ = dolfin_adjoint_str
backend.Function.interpolate = dolfin_adjoint_interpolate

if hasattr(backend.Function, 'sub'):
    backend.Function.sub = dolfin_adjoint_sub

def _check_and_extract_functions(e, linear_comb=None, scalar_weight=1.0,
                                 multi_index=None):
    """
    Utility func for extracting Functions and scalars in linear
    combinations of Functions
    """
    from ufl.algebra import Sum, Product, Division
    from ufl.classes import ComponentTensor
    linear_comb = linear_comb or []

    # First check u
    if isinstance(e, backend.Function):
        linear_comb.append((e, scalar_weight))
        return linear_comb

    # Second check a*u*b, u/a/b, a*u/b where a and b are scalars
    elif isinstance(e, (Product, Division)):
        linear_comb = _check_mul_and_division(e, linear_comb, scalar_weight, multi_index)
        return linear_comb

    # Third check a*u*b, u/a/b, a*u/b where a and b are scalars and u is a Tensor
    elif isinstance(e, ComponentTensor):
        e, multi_index = e.operands()
        linear_comb = _check_mul_and_division(e, linear_comb, scalar_weight, multi_index)
        return linear_comb

    # If not Product or Division we expect Sum
    elif isinstance(e, Sum):
        for op in e.operands():
            linear_comb = _check_and_extract_functions(op, linear_comb, \
                                                       scalar_weight, multi_index)

    else:
        _assign_error()

    return linear_comb

def _check_and_contract_linear_comb(expr, self, multi_index=None):
    """
    Utility func for checking and contracting linear combinations of
    Functions
    """
    linear_comb = _check_and_extract_functions(expr, multi_index=multi_index)
    funcs = []
    weights = []
    funcspace = None
    for func, weight in linear_comb:
        funcspace = funcspace or func.function_space()
        if func not in funcspace:
            _assign_error()
        try:
            # Check if the exact same Function is already present
            ind = funcs.index(func)
            weights[ind] += weight
        except:
            funcs.append(func)
            weights.append(weight)

    # Check that rhs does not include self
    for ind, func in enumerate(funcs):
        if func == self:
            # If so make a copy
            funcs[ind] = self.copy(deepcopy=True)
            break

    return zip(funcs, weights)

def _check_mul_and_division(e, linear_comb, scalar_weight=1.0, multi_index=None):
    """
    Utility func for checking division and multiplication of a Function
    with scalars in linear combinations of Functions
    """
    from ufl.constantvalue import ScalarValue
    from ufl.classes import ComponentTensor, MultiIndex, Indexed
    from ufl.algebra import Division, Product, Sum
    #ops = e.operands()

    # FIXME: What should be checked!?
    same_multi_index = lambda x, y: len(x.free_indices()) == len(y.free_indices()) \
                and x.index_dimensions().values() == y.index_dimensions().values()

    assert(isinstance(scalar_weight, float))

    # Split passed expression into scalar and expr
    if isinstance(e, Product):
        for i, op in enumerate(e.operands()):
            if isinstance(op, ScalarValue) or \
                   (isinstance(op, backend.Constant) and op.value_size()==1):
                scalar = op
                expr = e.operands()[1-i]
                break
        else:
            _assign_error()

        scalar_weight *= float(scalar)
    elif isinstance(e, Division):
        expr, scalar = e.operands()
        if not (isinstance(scalar, ScalarValue) or \
                isinstance(scalar, backend.Constant) and scalar.value_rank()==1):
            _assign_error()
        scalar_weight /= float(scalar)
    else:
        _assign_error()

    # If a CoefficientTensor is passed we expect the expr to be either a
    # Function or another ComponentTensor, where the latter wil result
    # in a recursive call
    if multi_index:
        assert(isinstance(multi_index, MultiIndex))
        assert(isinstance(expr, Indexed))

        # Unpack Indexed and check equality with passed multi_index
        expr, multi_index2 = expr.operands()
        assert(isinstance(multi_index2, MultiIndex))
        if not same_multi_index(multi_index, multi_index2):
            _assign_error()

    if isinstance(expr, backend.Function):
        linear_comb.append((expr, scalar_weight))

    elif isinstance(expr, (ComponentTensor, Product, Division, Sum)):
        # If componentTensor we need to unpack the MultiIndices
        if isinstance(expr, ComponentTensor):
            expr, multi_index = expr.operands()
            if not same_multi_index(multi_index, multi_index2):
                _error()

        if isinstance(expr, (Product, Division)):
            linear_comb = _check_mul_and_division(expr, linear_comb, \
                                                  scalar_weight, multi_index)
        elif isinstance(expr, Sum):
            linear_comb = _check_and_extract_functions(expr, linear_comb, \
                                                       scalar_weight, multi_index)
        else:
            _assign_error()
    else:
        _assign_error()

    return linear_comb

def _assign_error():
    assert False

class LinComRHS(libadjoint.RHS):
    def __init__(self, functions, weights, fn_space):
        self.fn_space = fn_space
        self.functions = functions
        self.weights = weights
        self.deps = [adjglobals.adj_variables[function] for function in functions]

    def __call__(self, dependencies, values):
        # Want to write
        # expr = sum(weight*value.data for (weight, value) in zip(self.weights, values))
        agh = [weight*value.data for (weight, value) in zip(self.weights, values)]
        expr = sum(agh[1:], agh[0])
        out = backend.Function(self.fn_space)
        out.assign(expr)
        return adjlinalg.Vector(out)

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        idx = dependencies.index(variable)

        # If you want to apply boundary conditions symmetrically in the adjoint
        # -- and you often do --
        # then we need to have a UFL representation of all the terms in the adjoint equation.
        # However!
        # Since UFL cannot represent the identity map,
        # we need to find an f such that when
        # assemble(inner(f, v)*dx)
        # we get the contraction_vector.data back.
        # This involves inverting a mass matrix.

        if backend.parameters["adjoint"]["symmetric_bcs"] and backend.__version__ <= '1.2.0':
            backend.info_red("Warning: symmetric BC application requested but unavailable in dolfin <= 1.2.0.")

        if backend.parameters["adjoint"]["symmetric_bcs"] and backend.__version__ > '1.2.0':

            V = contraction_vector.data.function_space()
            v = backend.TestFunction(V)

            if str(V) not in adjglobals.fsp_lu:
                u = backend.TrialFunction(V)
                A = backend.assemble(backend.inner(u, v)*backend.dx)
                lusolver = backend.LUSolver(A, "mumps")
                lusolver.parameters["symmetric"] = True
                lusolver.parameters["reuse_factorization"] = True
                adjglobals.fsp_lu[str(V)] = lusolver
            else:
                lusolver = adjglobals.fsp_lu[str(V)]

            riesz = backend.Function(V)
            lusolver.solve(riesz.vector(), self.weights[idx] * contraction_vector.data.vector())
            out = (backend.inner(riesz, v)*backend.dx)
        else:
            out = backend.Function(self.fn_space)
            out.assign(self.weights[idx] * contraction_vector.data)

        return adjlinalg.Vector(out)

    def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action):
        return None

    def dependencies(self):
        return self.deps

    def coefficients(self):
        return self.functions

    def __str__(self):
        return "LinComRHS(%s)" % str(self.dep)
