import backend
import copy
import utils
import caching

backend_assemble = backend.assemble
def assemble(*args, **kwargs):
    """When a form is assembled, the information about its nonlinear dependencies is lost,
    and it is no longer easy to manipulate. Therefore, dolfin_adjoint overloads the :py:func:`dolfin.assemble`
    function to *attach the form to the assembled object*. This lets the automatic annotation work,
    even when the user calls the lower-level :py:data:`solve(A, x, b)`.
    """
    form = args[0]
    caching.assembled_fwd_forms.add(form)
    cache = kwargs.pop("cache", False)

    to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

    output = backend_assemble(*args, **kwargs)
    if not isinstance(output, float) and to_annotate:
        output.form = form
        output.assemble_system = False

    if cache:
        caching.assembled_adj_forms[form] = output

    return output

if hasattr(backend, 'PeriodicBC'):
    periodic_bc_apply = backend.PeriodicBC.apply
    def adjoint_periodic_bc_apply(self, *args, **kwargs):
        for arg in args:
            if not hasattr(arg, 'bcs'):
                arg.bcs = []
            arg.bcs.append(self)
        return periodic_bc_apply(self, *args, **kwargs)
    backend.PeriodicBC.apply = adjoint_periodic_bc_apply

if hasattr(backend, 'DirichletBC'):
    dirichlet_bc_apply = backend.DirichletBC.apply
    def adjoint_dirichlet_bc_apply(self, *args, **kwargs):
        for arg in args:
            if not hasattr(arg, 'bcs'):
                arg.bcs = []
            arg.bcs.append(self)
        return dirichlet_bc_apply(self, *args, **kwargs)
    backend.DirichletBC.apply = adjoint_dirichlet_bc_apply

function_vector = backend.Function.vector
def adjoint_function_vector(self):
    vec = function_vector(self)
    vec.function = self
    return vec
backend.Function.vector = adjoint_function_vector

def assemble_system(*args, **kwargs):
    """When a form is assembled, the information about its nonlinear dependencies is lost,
    and it is no longer easy to manipulate. Therefore, dolfin_adjoint overloads the :py:func:`dolfin.assemble_system`
    function to *attach the form to the assembled object*. This lets the automatic annotation work,
    even when the user calls the lower-level :py:data:`solve(A, x, b)`.
    """
    lhs = args[0]
    rhs = args[1]
    caching.assembled_fwd_forms.add(lhs)
    caching.assembled_fwd_forms.add(rhs)

    cache = kwargs.pop("cache", False)

    if 'bcs' in kwargs:
        bcs = kwargs['bcs']
    elif len(args) > 2:
        bcs = args[2]
    else:
        bcs = []

    if not isinstance(bcs, list):
        bcs = [bcs]

    (lhs_out, rhs_out) = backend.assemble_system(*args, **kwargs)

    to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
    if to_annotate:
        lhs_out.form = lhs
        lhs_out.bcs = bcs
        lhs_out.assemble_system = True
        rhs_out.form = rhs
        rhs_out.bcs = bcs
        rhs_out.assemble_system = True

    if cache:
        caching.assembled_adj_forms[lhs] = lhs_out
        caching.assembled_adj_forms[rhs] = rhs_out

    return (lhs_out, rhs_out)
