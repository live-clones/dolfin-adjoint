"""
Implementation of Burger's equation with nonlinear solve in each
timestep and a functional integrating over time
"""

from __future__ import print_function
import sys

from firedrake import *
from firedrake_adjoint import *

firedrake.parameters["adjoint"]["record_all"] = True
firedrake.parameters["adjoint"]["fussy_replay"] = False

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    j = 0
    j += 0.5*float(timestep)*assemble(u_*u_*u_*u_*dx)

    if annotate:
        adjointer.time.start(0)

    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight*float(timestep)*assemble(u_*u_*u_*u_*dx)
        if annotate:
            adj_inc_timestep(time=t, finished=t>end)

    return j, u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    ic_copy = ic.copy(deepcopy=True, annotate=False)
    j, forward = main(ic, annotate=True)
    forward_copy = forward.copy(deepcopy=True, annotate=False)
    ic = forward
    ic.vector()[:] = ic_copy.vector()

    print("Running forward replay .... ")
    replay_dolfin(forget=False)

    print("Running adjoint ... ")

    J = Functional(forward*forward*forward*forward*dx*dt)
    m = Control(ic)
    Jm = j
    dJdm = compute_gradient(J, m, forget=False)
    HJm  = hessian(J, m)

    def Jfunc(ic):
        j, forward = main(ic, annotate=False)
        return j

    minconv = taylor_test(Jfunc, m, Jm, dJdm, HJm=HJm, seed=5.0e-4, perturbation_direction=interpolate(Expression("x[0]", degree=1), V))
    assert minconv > 2.7
