import backend
import ufl
from solving import solve, annotate as solving_annotate, do_checkpoint, register_initial_conditions
import libadjoint
import adjlinalg
import adjglobals
import utils

def register_assign(new, old, op=None):

    if not isinstance(old, backend.Function):
        assert op is not None

    fn_space = new.function_space()
    identity_block = utils.get_identity_block(fn_space)
    dep = adjglobals.adj_variables.next(new)

    if backend.parameters["adjoint"]["record_all"] and isinstance(old, backend.Function):
        adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(old)))

    rhs = IdentityRHS(old, fn_space, op)
    register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)
    initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
    cs = adjglobals.adjointer.register_equation(initial_eq)

    do_checkpoint(cs, dep, rhs)

class IdentityRHS(libadjoint.RHS):
    def __init__(self, var, fn_space, op):
        self.var = var
        self.fn_space = fn_space
        self.op = op
        if isinstance(var, backend.Function):
            self.dep = adjglobals.adj_variables[var]

    def __call__(self, dependencies, values):
        if len(values) > 0:
            return adjlinalg.Vector(values[0].data)
        else:
            return adjlinalg.Vector(self.op(self.var, self.fn_space))

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

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
            lusolver.solve(riesz.vector(), contraction_vector.data.vector())
            return adjlinalg.Vector(backend.inner(riesz, v)*backend.dx)
        else:
            return adjlinalg.Vector(contraction_vector.data)

    def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action):
        return None

    def dependencies(self):
        if isinstance(self.var, backend.Function):
            return [self.dep]
        else:
            return []

    def coefficients(self):
        if isinstance(self.var, backend.Function):
            return [self.var]
        else:
            return []

    def __str__(self):
        return "AssignIdentity(%s)" % str(self.dep)
