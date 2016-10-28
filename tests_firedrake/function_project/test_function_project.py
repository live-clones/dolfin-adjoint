from firedrake import *
from firedrake_adjoint import *
import numpy


def test_function_project():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, "CG", 1)

    c = Function(V, name="Control")
    c.project(Constant(1), annotate=False)

    u = Function(V, name="Projected")
    u.project(c, annotate=True)

    J = Functional(u**2*dx)
    ctrl = Control(c)
    Jhat = ReducedFunctional(J, ctrl)
    conv = Jhat.taylor_test(c)
    assert numpy.allclose(conv, 2)
