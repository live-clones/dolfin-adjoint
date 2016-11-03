
import backend
import libadjoint

from . import solving
from . import assignment
from . import expressions
from . import adjrhs

import hashlib
import random

def annotate_split(bigfn, idx, smallfn, bcs):
    fn_space = smallfn.function_space().collapse()
    test = backend.TestFunction(fn_space)
    trial = backend.TrialFunction(fn_space)
    eq_lhs = backend.inner(test, trial)*backend.dx

    diag_name = "Split:%s:" % idx + hashlib.md5(str(eq_lhs) + "split" + str(smallfn) + str(bigfn) + str(idx) + str(random.random())).hexdigest()

    diag_deps = []
    diag_block = libadjoint.Block(diag_name, dependencies=diag_deps, test_hermitian=backend.parameters["adjoint"]["test_hermitian"], test_derivative=backend.parameters["adjoint"]["test_derivative"])

    solving.register_initial_conditions([(bigfn, adjglobals.adj_variables[bigfn])], linear=True, var=None)

    var = adjglobals.adj_variables.next(smallfn)
    frozen_expressions_dict = expressions.freeze_dict()

    def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
        '''This callback must conform to the libadjoint Python block assembly
        interface. It returns either the form or its transpose, depending on
        the value of the logical hermitian.'''

        assert coefficient == 1

        expressions.update_expressions(frozen_expressions_dict)
        value_coeffs=[v.data for v in values]
        eq_l = eq_lhs

        if hermitian:
            adjoint_bcs = [utils.homogenize(bc) for bc in bcs if isinstance(bc, backend.DirichletBC)] + [bc for bc in bcs if not isinstance(bc, backend.DirichletBC)]
            if len(adjoint_bcs) == 0: adjoint_bcs = None
            return (adjlinalg.Matrix(backend.adjoint(eq_l), bcs=adjoint_bcs), adjlinalg.Vector(None, fn_space=fn_space))
        else:
            return (adjlinalg.Matrix(eq_l, bcs=bcs), adjlinalg.Vector(None, fn_space=fn_space))
    diag_block.assemble = diag_assembly_cb

    rhs = SplitRHS(test, bigfn, idx)

    eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)

    cs = adjglobals.adjointer.register_equation(eqn)
    solving.do_checkpoint(cs, var, rhs)

    if backend.parameters["adjoint"]["fussy_replay"]:
        mass = eq_lhs
        smallfn_massed = backend.Function(fn_space)
        backend.solve(mass == backend.action(mass, smallfn), smallfn_massed)
        assert False, "No idea how to assign to a subfunction yet .. "
        #assignment.dolfin_assign(bigfn, smallfn_massed)

    if backend.parameters["adjoint"]["record_all"]:
        smallfn_record = backend.Function(fn_space)
        assignment.dolfin_assign(smallfn_record, smallfn)
        adjglobals.adjointer.record_variable(var, libadjoint.MemoryStorage(adjlinalg.Vector(smallfn_record)))

class SplitRHS(adjrhs.RHS):
    def __init__(self, test, function, index):
        self.test = test
        self.function = function
        self.idx = index
        self.deps = [adjglobals.adj_variables[function]]
        self.coeffs = [function]

    def __call__(self, dependencies, values):
        fn = Function_split(values[0].data, deepcopy=True)[self.idx]
        return adjlinalg.Vector(backend.inner(self.test, fn)*backend.dx)

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        if not hermitian:
            fn = Function_split(contraction_vector)[self.idx]
            action = backend.inner(self.test, fn)
        else:
            bigtest = backend.TestFunction(self.function.function_space())
            outfn   = backend.Function(self.function.function_space())

            # DOLFIN is a bit annoying when it comes to splits. Actually, it is very annoying.
            # You can't do anything like
            # outfn[idx].vector()[:] = values_I_want_to_assign_to_outfn[idx]
            # or
            # fn = outfn.split()[idx]; fn.vector()[:] = values_I_want_to_assign_to_outfn[idx]
            # for whatever reason
            assert False, "No idea how to assign to a subfunction yet .. "
            assignment.dolfin_assign(outfn, contraction_vector.data)

            action = backend.inner(bigtest, outfn)*backend.dx

        return adjlinalg.Vector(action)
