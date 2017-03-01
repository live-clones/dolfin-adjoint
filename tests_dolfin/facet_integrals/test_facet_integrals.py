from dolfin import *
from dolfin_adjoint import *
import pytest

@pytest.mark.xfail(reason="Not fixed yet, see bitbucket issue #71")
def test_facet_integrals():
    mesh = UnitIntervalMesh(2)
    W = FunctionSpace(mesh, "CG", 1)

    u = Function(W)
    g = project(Constant(1), W)
    u_ = TrialFunction(W)
    v = TestFunction(W)

    F = u_*v * dx - g*v*dx
    solve(lhs(F) == rhs(F), u)

    J = Functional(0.5 * inner(u, u) * dS)

    # Reduced functional with single control
    m = Control(g)

    Jhat = ReducedFunctional(J, m)
    assert Jhat.taylor_test(g, test_hessian=True) > 2.9
