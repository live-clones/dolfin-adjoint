import dolfin
from . import solving
from . import assembly
import libadjoint
from . import adjglobals
from . import adjlinalg
from . import misc
from . import utils

lu_solvers = []
adj_lu_solvers = []

def make_LUSolverMatrix(idx, reuse_factorization):
    class LUSolverMatrix(adjlinalg.Matrix):
        def solve(self, var, b):

            if reuse_factorization is False:
                return adjlinalg.Matrix.solve(self, var, b)

            if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
                bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, dolfin.DirichletBC)]
            else:
                bcs = self.bcs

            if var.type in ['ADJ_FORWARD', 'ADJ_TLM']:
                solver = lu_solvers[idx]
                if solver is None:
                    A = assembly.assemble(self.data); [bc.apply(A) for bc in bcs]
                    lu_solvers[idx] = LUSolver(A)
                    lu_solvers[idx].parameters["reuse_factorization"] = True
                solver = lu_solvers[idx]

            else:
                if adj_lu_solvers[idx] is None:
                    A = assembly.assemble(self.data); [bc.apply(A) for bc in bcs]
                    adj_lu_solvers[idx] = LUSolver(A)
                    adj_lu_solvers[idx].parameters["reuse_factorization"] = True

                solver = adj_lu_solvers[idx]

            x = adjlinalg.Vector(dolfin.Function(self.test_function().function_space()))

            if b.data is None:
                # This means we didn't get any contribution on the RHS of the adjoint system. This could be that the
                # simulation ran further ahead than when the functional was evaluated, or it could be that the
                # functional is set up incorrectly.
                dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
            else:
                if isinstance(b.data, dolfin.Function):
                    b_vec = b.data.vector().copy()
                else:
                    b_vec = dolfin.assemble(b.data)

                [bc.apply(b_vec) for bc in bcs]
                solver.solve(x.data.vector(), b_vec, annotate=False)

            return x
    return LUSolverMatrix

class LUSolver(dolfin.LUSolver):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, *args):
        self.__global_list_idx__ = None

        if len(args) > 0:
            self.matrix = args[0]

        dolfin.LUSolver.__init__(self, *args)

    def set_operator(self, operator):
        self.matrix = operator
        dolfin.LUSolver.set_operator(self, operator)

    def solve(self, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

        to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

        if to_annotate:
            if len(args) == 2:
                try:
                    A = self.matrix.form
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your matrix A has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    self.op_bcs = self.matrix.bcs
                except AttributeError:
                    self.op_bcs = []

                try:
                    x = args[0].function
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

                try:
                    b = args[1].form
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    eq_bcs = misc.uniq(self.op_bcs + args[1].bcs)
                except AttributeError:
                    eq_bcs = self.op_bcs

            elif len(args) == 3:
                A = args[0].form
                try:
                    x = args[1].function
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

                try:
                    self.op_bcs = A.bcs
                except AttributeError:
                    self.op_bcs = []

                try:
                    b = args[2].form
                except AttributeError:
                    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    eq_bcs = misc.uniq(self.op_bcs + args[2].bcs)
                except AttributeError:
                    eq_bcs = self.op_bcs

            else:
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs("LUSolver.solve() must be called with either (A, x, b) or (x, b).")

            if self.parameters["reuse_factorization"] and self.__global_list_idx__ is None:
                self.__global_list_idx__ = len(lu_solvers)
                lu_solvers.append(self)
                adj_lu_solvers.append(None)

            solving.annotate(A == b, x, eq_bcs, solver_parameters={"linear_solver": "lu"}, matrix_class=make_LUSolverMatrix(self.__global_list_idx__, self.parameters["reuse_factorization"]))

        out = dolfin.LUSolver.solve(self, *args, **kwargs)

        if to_annotate:
            if dolfin.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[x], libadjoint.MemoryStorage(adjlinalg.Vector(x)))

        return out
