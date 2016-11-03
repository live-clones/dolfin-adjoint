import dolfin
from . import solving
import libadjoint
from . import adjglobals
from . import adjlinalg
from . import utils

class NewtonSolver(dolfin.NewtonSolver):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def solve(self, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

        to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

        if to_annotate:
            factory = args[0]
            vec = args[1]
            b = dolfin.as_backend_type(vec).__class__()

            factory.F(b=b, x=vec)

            F = b.form
            bcs = b.bcs

            u = vec.function
            var = adjglobals.adj_variables[u]

            solving.annotate(F == 0, u, bcs, solver_parameters={"newton_solver": self.parameters.to_dict()})

        newargs = [self] + list(args)
        out = dolfin.NewtonSolver.solve(*newargs, **kwargs)

        if to_annotate and dolfin.parameters["adjoint"]["record_all"]:
            adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))

        return out
