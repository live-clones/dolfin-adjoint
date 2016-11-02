"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
from firedrake import *
from firedrake_adjoint import *


@pytest.fixture
def V():
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)
    return V


def main(ic, annotate=annotate):

    V = ic.function_space()

    n = V.mesh().num_cells()

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    u_ = ic.copy(deepcopy=True, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, (1, 2))

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_


def test_burgers_newton(V):
    ic = project(Expression("sin(2*pi*x[0])"), V)

    forward = main(ic, annotate=True)

    print "Running forward replay .... "
    replay_dolfin(forget=False)
    print "Running adjoint ... "

    J = Functional(forward*forward*dx*dt[FINISH_TIME] + forward*forward*dx*dt[START_TIME])
    Jic = assemble(forward*forward*dx + ic*ic*dx)
    dJdic = compute_gradient(J, FunctionControl("Velocity"), forget=False)

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx + ic*ic*dx)

    HJic = hessian(J, FunctionControl("Velocity"), warn=False)

    minconv = taylor_test(Jfunc, FunctionControl("Velocity"), Jic, dJdic, HJm=HJic, seed=1.0e-3, perturbation_direction=interpolate(Expression("cos(x[0])"), V, annotate=False))
    assert minconv > 2.7
