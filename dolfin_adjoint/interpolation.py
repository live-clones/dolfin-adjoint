import solving
import backend
import libadjoint
import ufl
import adjglobals
import adjlinalg
import utils

def interpolate(v, V, annotate=None, name=None):
    '''The interpolate call changes Function data, and so it too must be annotated so that the
    adjoint and tangent linear models may be constructed automatically by libadjoint.

    To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
    cases where the interpolation is known to be irrelevant or diagnostic for the purposes of the adjoint
    computation (such as interpolating fields to other function spaces for the purposes of
    visualisation).'''

    out = backend.interpolate(v, V)
    out = utils.function_to_da_function(out)
    if name is not None:
        out.adj_name = name

    to_annotate = utils.to_annotate(annotate)

    if isinstance(v, backend.Function) and to_annotate:
        rhsdep = adjglobals.adj_variables[v]
        if adjglobals.adjointer.variable_known(rhsdep):
            rhs = InterpolateRHS(v, V)
            identity_block = utils.get_identity_block(V)

            solving.register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)

            dep = adjglobals.adj_variables.next(out)

            if backend.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(out)))

            initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
            cs = adjglobals.adjointer.register_equation(initial_eq)

            solving.do_checkpoint(cs, dep, rhs)

    return out

class InterpolateRHS(libadjoint.RHS):
    def __init__(self, v, V):
        self.v = v
        self.V = V
        self.dep = adjglobals.adj_variables[v]

    def __call__(self, dependencies, values):
        return adjlinalg.Vector(backend.interpolate(values[0].data, self.V))

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        if not hermitian:
            return adjlinalg.Vector(backend.interpolate(contraction_vector.data, self.V))
        else:
#      For future reference, the transpose action of the interpolation operator
#      is (in pseudocode!):
#
#      for target_dof in target:
#        figure out what element it lives in, to compute src_dofs
#        for src_dof in src_dofs:
#          basis = the value of the basis function of src_dof at the node of target_dof
#
#          # all of the above is exactly the same as the forward interpolation.
#          # forward interpolation would do:
#          # target_coefficients[target_dof] += basis * src_coefficients[src_dof]
#
#          # but the adjoint action is:
#          src_coefficients[src_dof] += basis * target_coefficients[target_dof]

            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Can't transpose an interpolation operator yet, sorry!")

    def dependencies(self):
        return [self.dep]

    def coefficients(self):
        return [self.v]

    def __str__(self):
        return "InterpolateRHS(%s, %s)" % (str(self.v), str(self.V))
