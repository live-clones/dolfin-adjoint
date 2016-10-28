from firedrake import *
from firedrake_adjoint import *
import pytest


@pytest.fixture
def mesh():
    return UnitSquareMesh(4, 4)


@pytest.fixture
def V3(mesh):
    return FunctionSpace(mesh, "CG", 3)


@pytest.fixture
def V2(mesh):
    return FunctionSpace(mesh, "CG", 2)


def main(ic, V2, annotate=False):
    return project(ic, V2, annotate=annotate)


def test_projection(V2, V3):
    firedrake.parameters["adjoint"]["record_all"] = True

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V3)
    soln = main(ic, V2, annotate=True)

    adj_html("projection_forward.html", "forward")
    assert replay_dolfin(tol=1e-12, stop=True)

    J = Functional(soln*soln*dx*dt[FINISH_TIME])
    Jic = assemble(soln*soln*dx)
    dJdic = compute_gradient(J, FunctionControl(ic), forget=False)

    def J(ic):
        soln = main(ic, V2, annotate=False)
        return assemble(soln*soln*dx)

    minconv = taylor_test(J, FunctionControl(ic), Jic, dJdic)
    assert minconv > 1.9
