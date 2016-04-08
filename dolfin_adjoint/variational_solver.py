import backend
import solving
import libadjoint
import adjglobals
import adjlinalg
import utils
import compatibility

class NonlinearVariationalProblem(backend.NonlinearVariationalProblem):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, F, u, bcs=None, J=None, *args, **kwargs):
        self.F = F
        self.u = u
        self.bcs = bcs
        self.J = J
        backend.NonlinearVariationalProblem.__init__(self, F, u, bcs, J, *args, **kwargs)

class NonlinearVariationalSolver(backend.NonlinearVariationalSolver):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, problem, *args, **kwargs):
        super(NonlinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self.problem = problem

    def solve(self, annotate=None):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

        annotate = utils.to_annotate(annotate)

        if annotate:
            problem = self.problem
            solving.annotate(problem.F == 0, problem.u, problem.bcs, J=problem.J, solver_parameters=compatibility.to_dict(self.parameters))

        out = backend.NonlinearVariationalSolver.solve(self)

        if annotate and backend.parameters["adjoint"]["record_all"]:
            adjglobals.adjointer.record_variable(adjglobals.adj_variables[self.problem.u], libadjoint.MemoryStorage(adjlinalg.Vector(self.problem.u)))

        return out

class LinearVariationalProblem(backend.LinearVariationalProblem):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, a, L, u, bcs=None, *args, **kwargs):
        backend.LinearVariationalProblem.__init__(self, a, L, u, bcs, *args, **kwargs)
        self.a = a
        self.L = L
        self.u = u
        self.bcs = bcs

class LinearVariationalSolver(backend.LinearVariationalSolver):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, problem, *args, **kwargs):
        super(LinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self.problem = problem

    def solve(self, annotate=None):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

        annotate = utils.to_annotate(annotate)

        if annotate:
            problem = self.problem
            solving.annotate(problem.a == problem.L, problem.u, problem.bcs, solver_parameters=compatibility.to_dict(self.parameters))

        out = backend.LinearVariationalSolver.solve(self)

        if annotate and backend.parameters["adjoint"]["record_all"]:
            adjglobals.adjointer.record_variable(adjglobals.adj_variables[self.problem.u], libadjoint.MemoryStorage(adjlinalg.Vector(self.problem.u)))

        return out
