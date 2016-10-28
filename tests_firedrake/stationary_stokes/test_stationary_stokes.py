from firedrake import *
from firedrake_adjoint import *
import pytest


length = 10


@pytest.fixture
def W():
    n = 3
    mesh = RectangleMesh(2**n, 2**n, length, 1)

    P1 = FiniteElement("CG", cell=mesh.ufl_cell(), degree=1)
    B = FiniteElement("B", cell=mesh.ufl_cell(), degree=3)
    mini = P1 + B
    V = VectorFunctionSpace(mesh, mini)
    P = FunctionSpace(mesh, 'CG', 1)
    return V*P


def model(s, W):
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx
    L = inner(s, v)*dx
    # No-slip velocity boundary condition on top and bottom,
    # y == 0 and y == 1
    noslip = Constant((0, 0))
    bc0 = DirichletBC(W[0], noslip, (3, 4))

    # Parabolic inflow y(1-y) at x = 0 in positive x direction
    inflow = Expression(("x[1]*(1 - x[1])", "0.0"))
    bc1 = DirichletBC(W[0], inflow, 1)

    # Zero pressure at outlow at x = 1
    bc2 = DirichletBC(W[1], 0.0, 2)

    bcs = [bc0, bc1, bc2]

    w = Function(W)

    u, p = w.split()

    solve(a == L, w, bcs=bcs, solver_parameters={"pc_type": "lu",
                                                 "pc_factor_shift_type": "nonzero",
                                                 "ksp_type": "preonly",
                                                 "mat_type": "aij"})

    x, y = SpatialCoordinate(W.mesh())

    expect = Function(W, name="expect")

    expect.sub(0).project(as_vector([y*(1 - y), 0]))
    expect.sub(1).interpolate(2*(length - x))

    return assemble(inner(expect - w, expect - w)*dx), w, expect


def test_stationary_stokes(W):
    V, P = W.split()
    s = Function(V, name="s")
    s.assign(0)

    print "Running forward model"
    j, u, f = model(s, W)
    print "Replaying forward model"
    assert replay_dolfin(tol=1e-5, stop=True)
    J = Functional(inner(u - f, u - f) * dx * dt[FINISH_TIME])
    m = FunctionControl(s)

    print "Running adjoint model"
    dJdm = compute_gradient(J, m, forget=None)

    parameters["adjoint"]["stop_annotating"] = True

    Jhat = lambda s: model(s, W)[0]
    conv_rate = taylor_test(Jhat, m, j, dJdm, seed=1e-3)
    assert conv_rate > 1.9
