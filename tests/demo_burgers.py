"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

# Last changed: 2012-01-09

from dolfin import *

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(n):

    mesh = UnitInterval(n)
    V = FunctionSpace(mesh, "CG", 2)

    u_ = project(Expression("sin(2*pi*x[0])"),  V)
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u_*grad(u)*v + nu*grad(u)*grad(v))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    u = Function(V)
    while (t <= end):
        solve(a == L, u, bc)

        u_.assign(u)

        t += float(timestep)
        plot(u)

    interactive()

if __name__ == "__main__":

    main(100)


