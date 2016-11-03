from __future__ import print_function
import ufl
import backend

import libadjoint

from . import solving
from . import expressions
from . import adjrhs
from . import adjlinalg
from . import adjglobals

import hashlib
import copy
import random

def down_cast(*args, **kwargs):
    """When a form is assembled, the information about its nonlinear dependencies is lost,
    and it is no longer easy to manipulate. Therefore, dolfin_adjoint overloads the :py:func:`dolfin.down_cast`
    function to *attach the form to the returned object*. This lets the automatic annotation work,
    even when the user calls the lower-level :py:data:`solve(A, x, b)`.
    """
    dc = backend.down_cast(*args, **kwargs)

    if hasattr(args[0], 'form'):
        dc.form = args[0].form

    if hasattr(args[0], 'function'):
        dc.function = args[0].function

    if hasattr(args[0], 'bcs'):
        dc.bcs = args[0].bcs

    return dc

class MatrixFree(adjlinalg.Matrix):
    def __init__(self, *args, **kwargs):
        self.fn_space = kwargs['fn_space']
        del kwargs['fn_space']

        self.operators = kwargs['operators']
        del kwargs['operators']

        self.parameters = kwargs['parameters']
        del kwargs['parameters']

        adjlinalg.Matrix.__init__(self, *args, **kwargs)

    def solve(self, var, b):
        timer = backend.Timer("Matrix-free solver")
        solver = backend.PETScKrylovSolver(*self.solver_parameters)
        solver.parameters.update(self.parameters)

        x = backend.Function(self.fn_space)
        if b.data is None:
            backend.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
            return adjlinalg.Vector(x)

        if isinstance(b.data, backend.Function):
            rhs = b.data.vector().copy()
        else:
            rhs = backend.assemble(b.data)

        if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
            self.bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, backend.cpp.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, backend.DirichletBC)]

        for bc in self.bcs:
            bc.apply(rhs)

        if self.operators[1] is not None: # we have a user-supplied preconditioner
            solver.set_operators(self.data, self.operators[1])
            solver.solve(backend.down_cast(x.vector()), backend.down_cast(rhs))
        else:
            solver.solve(self.data, backend.down_cast(x.vector()), backend.down_cast(rhs))

        timer.stop()
        return adjlinalg.Vector(x)

    def axpy(self, alpha, x):
        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Can't add to a matrix-free matrix .. ")

class AdjointPETScKrylovSolver(backend.PETScKrylovSolver):
    def __init__(self, *args):
        backend.PETScKrylovSolver.__init__(self, *args)
        self.solver_parameters = args

        self.operators = (None, None)

    def set_operators(self, A, P):
        backend.PETScKrylovSolver.set_operators(self, A, P)
        self.operators = (A, P)

    def set_operator(self, A):
        backend.PETScKrylovSolver.set_operator(self, A)
        self.operators = (A, self.operators[1])

    def solve(self, *args, **kwargs):

        timer = backend.Timer("Matrix-free solver")

        annotate = True
        if "annotate" in kwargs:
            annotate = kwargs["annotate"]
            del kwargs["annotate"]

        if len(args) == 3:
            A = args[0]
            x = args[1]
            b = args[2]
        elif len(args) == 2:
            A = self.operators[0]
            x = args[0]
            b = args[1]

        if annotate:
            if not isinstance(A, AdjointKrylovMatrix):
                try:
                    A = AdjointKrylovMatrix(A.form)
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your A has to either be an AdjointKrylovMatrix or have been assembled after backend_adjoint was imported.")

            if not hasattr(x, 'function'):
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your x has to come from code like down_cast(my_function.vector()).")

            if not hasattr(b, 'form'):
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your b has to have the .form attribute: was it assembled with from backend_adjoint import *?")

            if not hasattr(A, 'dependencies'):
                backend.info_red("A has no .dependencies method; assuming no nonlinear dependencies of the matrix-free operator.")
                coeffs = []
                dependencies = []
            else:
                coeffs = [coeff for coeff in A.dependencies() if hasattr(coeff, 'function_space')]
                dependencies = [adjglobals.adj_variables[coeff] for coeff in coeffs]

            if len(dependencies) > 0:
                assert hasattr(A, "set_dependencies"), "Need a set_dependencies method to replace your values, if you have nonlinear dependencies ... "

            rhs = adjrhs.RHS(b.form)

            diag_name = hashlib.md5(str(hash(A)) + str(random.random())).hexdigest()
            diag_block = libadjoint.Block(diag_name, dependencies=dependencies, test_hermitian=backend.parameters["adjoint"]["test_hermitian"], test_derivative=backend.parameters["adjoint"]["test_derivative"])

            solving.register_initial_conditions(zip(rhs.coefficients(), rhs.dependencies()) + zip(coeffs, dependencies), linear=False, var=None)

            var = adjglobals.adj_variables.next(x.function)

            frozen_expressions_dict = expressions.freeze_dict()
            frozen_parameters = self.parameters.to_dict()

            def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
                '''This callback must conform to the libadjoint Python block assembly
                interface. It returns either the form or its transpose, depending on
                the value of the logical hermitian.'''

                assert coefficient == 1

                expressions.update_expressions(frozen_expressions_dict)

                if len(dependencies) > 0:
                    A.set_dependencies(dependencies, [val.data for val in values])

                if hermitian:
                    A_transpose = A.hermitian()
                    return (MatrixFree(A_transpose, fn_space=x.function.function_space(), bcs=A_transpose.bcs,
                                       solver_parameters=self.solver_parameters,
                                       operators=transpose_operators(self.operators),
                                       parameters=frozen_parameters), adjlinalg.Vector(None, fn_space=x.function.function_space()))
                else:
                    return (MatrixFree(A, fn_space=x.function.function_space(), bcs=b.bcs,
                                       solver_parameters=self.solver_parameters,
                                       operators=self.operators,
                                       parameters=frozen_parameters), adjlinalg.Vector(None, fn_space=x.function.function_space()))
            diag_block.assemble = diag_assembly_cb

            def diag_action_cb(dependencies, values, hermitian, coefficient, input, context):
                expressions.update_expressions(frozen_expressions_dict)
                A.set_dependencies(dependencies, [val.data for val in values])

                if hermitian:
                    acting_mat = A.transpose()
                else:
                    acting_mat = A

                output_fn = backend.Function(input.data.function_space())
                acting_mat.mult(input.data.vector(), output_fn.vector())
                vec = output_fn.vector()
                for i in range(len(vec)):
                    vec[i] = coefficient * vec[i]

                return adjlinalg.Vector(output_fn)
            diag_block.action = diag_action_cb

            if len(dependencies) > 0:
                def derivative_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
                    expressions.update_expressions(frozen_expressions_dict)
                    A.set_dependencies(dependencies, [val.data for val in values])

                    action = A.derivative_action(values[dependencies.index(variable)].data, contraction_vector.data, hermitian, input.data, coefficient)
                    return adjlinalg.Vector(action)
                diag_block.derivative_action = derivative_action

            eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)
            cs = adjglobals.adjointer.register_equation(eqn)
            solving.do_checkpoint(cs, var, rhs)

        out = backend.PETScKrylovSolver.solve(self, *args)

        if annotate:
            if backend.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(var, libadjoint.MemoryStorage(adjlinalg.Vector(x.function)))

        timer.stop()

        return out

class AdjointKrylovMatrix(backend.PETScKrylovMatrix):
    def __init__(self, a, bcs=None):
        shapes = self.shape(a)
        backend.PETScKrylovMatrix.__init__(self, shapes[0], shapes[1])
        self.original_form = a
        self.current_form = a

        if bcs is None:
            self.bcs = []
        else:
            if isinstance(bcs, list):
                self.bcs = bcs
            else:
                self.bcs = [bcs]

    def shape(self, a):
        args = ufl.algorithms.extract_arguments(a)
        shapes = [arg.function_space().dim() for arg in args]
        return shapes

    def mult(self, *args):
        shapes = self.shape(self.current_form)
        y = backend.PETScVector(shapes[0])

        action_fn = backend.Function(ufl.algorithms.extract_arguments(self.current_form)[-1].function_space())
        action_vec = action_fn.vector()
        for i in range(len(args[0])):
            action_vec[i] = args[0][i]

        action_form = backend.action(self.current_form, action_fn)
        backend.assemble(action_form, tensor=y)

        for bc in self.bcs:
            bcvals = bc.get_boundary_values()
            for idx in bcvals:
                y[idx] = action_vec[idx]

        args[1].set_local(y.array())

    def dependencies(self):
        return ufl.algorithms.extract_coefficients(self.original_form)

    def set_dependencies(self, dependencies, values):
        replace_dict = dict(zip(self.dependencies(), values))
        self.current_form = backend.replace(self.original_form, replace_dict)

    def hermitian(self):
        adjoint_bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, backend.cpp.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, backend.DirichletBC)]
        return AdjointKrylovMatrix(backend.adjoint(self.original_form), bcs=adjoint_bcs)

    def derivative_action(self, variable, contraction_vector, hermitian, input, coefficient):
        deriv = backend.derivative(self.current_form, variable)
        args = ufl.algorithms.extract_arguments(deriv)
        deriv = backend.replace(deriv, {args[1]: contraction_vector})

        if hermitian:
            deriv = backend.adjoint(deriv)

        action = coefficient * backend.action(deriv, input)

        return action

def transpose_operators(operators):
    out = [None, None]

    for i in range(2):
        op = operators[i]

        if op is None:
            out[i] = None
        elif isinstance(op, backend.cpp.GenericMatrix):
            out[i] = op.__class__()
            backend.assemble(backend.adjoint(op.form), tensor=out[i])

            if hasattr(op, 'bcs'):
                adjoint_bcs = [utils.homogenize(bc) for bc in op.bcs if isinstance(bc, backend.cpp.DirichletBC)] + [bc for bc in op.bcs if not isinstance(bc, backend.DirichletBC)]
                [bc.apply(out[i]) for bc in adjoint_bcs]

        elif isinstance(op, backend.Form) or isinstance(op, ufl.form.Form):
            out[i] = backend.adjoint(op)

            if hasattr(op, 'bcs'):
                out[i].bcs = [utils.homogenize(bc) for bc in op.bcs if isinstance(bc, backend.cpp.DirichletBC)] + [bc for bc in op.bcs if not isinstance(bc, backend.DirichletBC)]

        elif isinstance(op, AdjointKrylovMatrix):
            pass

        else:
            print("op.__class__: ", op.__class__)
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to transpose anything else!")

    return out
