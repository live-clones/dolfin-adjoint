"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *
from math import ceil

n = 100
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

#dolfin.parameters["adjoint"]["record_all"] = True
#dolfin.parameters["adjoint"]["test_hermitian"] = (100, 1.0e-14)
#dolfin.parameters["adjoint"]["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):
    timestep = Constant(1.0/n)
    t = 0.0
    end = 0.5
    if annotate:
        adj_checkpointing('multistage', int(ceil(end/float(timestep))), 5, 10, verbose=True)

    u_ = ic.copy(deepcopy=True, annotate=annotate)
    u = TrialFunction(V)
    v = TestFunction(V)
    nu = Constant(0.0001)

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    u = Function(V)
    j = 0
    j += 0.5*float(timestep)*assemble(u_*u_*dx)
    if annotate:
        adjointer.time.start(t)

    while (t <= end):
        solve(a == L, u, bc, annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight*float(timestep)*assemble(u_*u_*dx)

        if annotate:
            adj_inc_timestep(time=t, finished=t>end)
        #plot(u)

    #interactive()
    return j, u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V, name="InitialCondition")
    j, forward = main(ic, annotate=True)

    J = Functional(forward*forward*dx*dt)
    m = Control(ic, value=ic)
    dJdm = compute_gradient(J, m)

    def Jhat(ic):
        j, forward = main(ic, annotate=False)
        return j

    minconv = taylor_test(Jhat, m, j, dJdm, seed=1.0e-3)
    assert minconv > 1.8
