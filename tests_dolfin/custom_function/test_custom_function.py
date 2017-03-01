from dolfin import *
from dolfin_adjoint import *
import libadjoint

class CustomDolfinAdjointFunction(object):

    def __call__(self, infunc, annotate=None):

        # Perform the forward action
        out = Function(infunc.function_space())
        out.vector()[:] = out.vector().array()**2

        # Annotate the operation on the dolfin-adjoint tape
        if utils.to_annotate(annotate):

            rhs = CustomDolfinAdjointFunctionRhs(infunc)
            out_dep = adjglobals.adj_variables.next(out)

            solving.register_initial_conditions(zip(rhs.coefficients(), rhs.dependencies()), linear=False)

            if parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(out_dep, libadjoint.MemoryStorage(adjlinalg.Vector(infunc)))

            identity = utils.get_identity_block(out.function_space())
            eq = libadjoint.Equation(out_dep, blocks=[identity], targets=[out_dep], rhs=rhs)
            cs = adjglobals.adjointer.register_equation(eq)

            solving.do_checkpoint(cs, out_dep, rhs)

        return out

class CustomDolfinAdjointFunctionRhs(libadjoint.RHS):
    def __init__(self, infunc):
        self.infunc = infunc
        self.deps = [adjglobals.adj_variables[self.infunc]]

    def __call__(self, dependencies, values):
        #from IPython import embed; embed()
        out = values[0].data.copy(deepcopy=True)
        out.vector()[:] = out.vector().array()**2
        
        return adjlinalg.Vector(out)

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        if hermitian:
            out = values[0].data.copy(deepcopy=True)
            out.vector()[:] = 2*out.vector().array()*contraction_vector.data.vector().array()
        else:
            raise NotImplementedError()
        
        return adjlinalg.Vector(out)

    def dependencies(self):
        # What does this custom function depend on?
        return self.deps 

    def coefficients(self):
        return [self.infunc]

    def __str__(self):
        return "CustomDolfinAdjointFunction"



def test_custom_function():
    mesh = UnitIntervalMesh(2)
    W = FunctionSpace(mesh, "CG", 1)

    g = project(Constant(2), W, name="in")

    func = CustomDolfinAdjointFunction()
    v = func(g)
    adj_html("forward.html", "forward")
    print v.vector().array()

    J = Functional(0.5 * inner(v, v) * dx)

    # Reduced functional with single control
    m = Control(g)

    Jhat = ReducedFunctional(J, m)
    assert Jhat.taylor_test(g) > 1.9
