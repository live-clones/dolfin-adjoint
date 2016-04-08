import sys

from dolfin import *
from dolfin_adjoint import *

n = 3
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

dolfin.parameters["adjoint"]["record_all"] = True

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="Velocity")
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)
    if annotate: assert hasattr(mass, 'form')

    advec = assemble(u_*u.dx(0)*v * dx)
    if annotate: assert hasattr(advec, 'form')

    rhs = assemble(inner(u_, v) * dx)
    if annotate: assert hasattr(rhs, 'form')

    L = mass + advec

    if annotate: assert hasattr(L, 'form')
    solve(L, u_.vector(), rhs, 'lu', annotate=annotate)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    forward = main(ic, annotate=True)
    adj_html("forward.html", "forward")
    print "Running adjoint ... "

    J = Functional(forward*forward*dx*dt[FINISH_TIME])
    m = FunctionControl("Velocity")
    Jm = assemble(forward*forward*dx)
    dJdm = compute_gradient(J, m, forget=False)

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm, seed=1.0e-6)
    assert minconv > 1.8
