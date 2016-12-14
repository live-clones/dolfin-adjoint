import random

from firedrake import *
from firedrake_adjoint import *
import sys

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 3)
firedrake.parameters["adjoint"]["record_all"] = True
a = Constant(2.0, name="a")

def main(ic, a, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)
    
    bc = DirichletBC(V, -1.0, "on_boundary")

    mass = inner(u, v)*dx
    rhs = a*action(mass, ic)
    soln = Function(V)
    da = Function(V)

    solve(mass == rhs, soln, bc, annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=1), V)
    soln = main(ic, a, annotate=True)

    J = Functional(soln*soln*dx*dt[FINISH_TIME])
    dJda = compute_gradient(J, ConstantControl("a"))

    def J(a):
        soln = main(ic, a, annotate=False)
        return assemble(soln*soln*dx)

    Ja = assemble(soln**2*dx)
    taylor_test(J, Control(a), Ja, dJda)
