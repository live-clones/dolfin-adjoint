from firedrake import *
from firedrake_adjoint import *
import pytest


@pytest.fixture
def V():
    # Create mesh and define function space
    n = 5
    mesh = UnitSquareMesh(2 ** n, 2 ** n)
    return FunctionSpace(mesh, "CG", 1)


def model(s, V):
    # Define variational problem
    x = Function(V, name="State")
    v = TestFunction(V)
    a = dot(v, x) * dx
    L = s * v * dx

    # Compute solution
    solve(a - L == 0, x)

    j = assemble(x**2 * dx)
    return j, x


def test_identity_nonlinear(V):
    s = Function(V, name="s")
    s.interpolate(Expression("1"))

    print "Running forward model"
    j, x = model(s, V)

    adj_html("forward.html", "forward")

    print "Replaying forward model"
    assert replay_dolfin(tol=0.0, stop=True)

    J = Functional(x**2*dx*dt[FINISH_TIME])
    m = FunctionControl(s)

    print "Running the adjoint model"
    for i in compute_adjoint(J, forget=None):
        pass

    print "Computing the gradient with the adjoint model"
    dJdm = compute_gradient(J, m, forget=None)

    parameters["adjoint"]["stop_annotating"] = True

    Jhat = lambda s: model(s, V)[0]
    conv_rate = taylor_test(Jhat, m, j, dJdm)
    assert conv_rate > 1.9
