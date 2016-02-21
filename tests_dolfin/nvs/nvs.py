from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.parameters["adjoint"]["record_all"] = True

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
f = interpolate(Expression("sin(pi*x[0])"), V, annotate = True, name = "ic")

def main(f, annotate=False):
    u = Function(V)
    a = u*v*dx
    L = f*v*dx
    F = a - L

    bcs = None

    problem = NonlinearVariationalProblem(F, u, bcs, J=derivative(F, u))
    solver = NonlinearVariationalSolver(problem)
    solver.solve(annotate=annotate)
    return u

u = main(f, annotate=True)
replay_dolfin()
grad = compute_gradient(Functional(u*u*dx*dt[FINISH_TIME] +
                        f*f*dx*dt[START_TIME]), Control(f), forget = False)

def J(f):
    u = main(f, annotate=False)
    return assemble(u*u*dx + f*f*dx)

Jf = assemble(u*u*dx + f*f*dx)
minconv = taylor_test(J, FunctionControl("ic"), Jf, grad)
if minconv < 1.9:
    sys.exit(1)
