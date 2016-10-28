""" Solves an optimisation problem with the Burgers equation as constraint """

import sys
from firedrake import *
from firedrake_adjoint import *
import pytest
try:
    import scipy                # noqa: F401
except ImportError:
    pass


@pytest.fixture
def V():
    n = 10
    mesh = UnitIntervalMesh(n)
    return FunctionSpace(mesh, "CG", 2)


def main(u, V, annotate=False):
    u_next = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    n = V.mesh().num_cells()
    timestep = Constant(1.0/n)

    def Dt(u_next, u, timestep):
        return (u_next - u)/timestep

    F = (Dt(u_next, u, timestep)*v +
         u_next*u_next.dx(0)*v + nu*u_next.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    adjointer.time.start(t)
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep(time=t, finished=(t > end))


@pytest.fixture(params=["L-BFGS-B",
                        "SLSQP",
                        "BFGS",
                        "COBYLA",
                        "TNC",
                        "Newton-CG",
                        "Nelder-Mead",
                        "CG"])
def method(request):
    return request.param


@pytest.fixture
def lb(V):
    return project(Expression("-1"), V)


@pytest.fixture
def options(method, lb):
    return {"L-BFGS-B": {"bounds": (lb, 1)},
            "SLSQP": {"bounds": (lb, 1)},
            "BFGS": {"bounds": None},
            "COBYLA": {"bounds": None, "rhobeg": 0.1},
            "TNC": {"bounds": None},
            "Newton-CG": {"bounds": None},
            "Nelder-Mead": {"bounds": None},
            "CG": {"bounds": None}}[method]


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Scipy optimization requires scipy library to be installed")
def test_optimization_scipy(V, method, options):
    ic = project(Expression("sin(2*pi*x[0])"), V, annotate=False)
    u = Function(ic, name='Velocity')

    J = Functional(u*u*dx*dt[FINISH_TIME])

    # Run the model once to create the annotation
    u.assign(ic, annotate=False)
    main(u, V, annotate=True)

    # Run the optimisation
    # Define the reduced funtional
    def derivative_cb(j, dj, m):
        print "j = %f, max(dj) = %f, max(m) = %f." % (j, dj.vector().max(),
                                                      m.vector().max())

    reduced_functional = ReducedFunctional(J, Control(u, value=ic),
                                           derivative_cb_post=derivative_cb)
    reduced_functional.taylor_test(ic)

    bounds = options.pop("bounds")
    # First solve with L-BFGS
    u_opt = minimize(reduced_functional, method="L-BFGS-B", bounds=bounds, tol=1e-10,
                     options=dict({"disp": True}))
    tol = 1e-9
    final_functional = reduced_functional(u_opt)
    assert final_functional <= tol

    # Now run again with the actual provided method
    u_opt = minimize(reduced_functional, method=method, bounds=bounds, tol=1e-10,
                     options=dict({"disp": True, "maxiter": 2}, **options))
    tol = 1e-9
    final_functional = reduced_functional(u_opt)
    assert final_functional <= tol
