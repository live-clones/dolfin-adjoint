from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.parameters["adjoint"]["record_all"] = True

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
f = interpolate(Expression("sin(pi*x[0])", degree=1), V, name = "initial condition")

def main(f, annotate=False):
    u = Function(V, name = "system state")
    w = TrialFunction(V)
    a = w*v*dx
    L = f*v*dx
    F = a - L

    bcs = None

    problem = LinearVariationalProblem(a, L, u, bcs)
    problem = LinearVariationalProblem(a, L, u)
    solver = LinearVariationalSolver(problem)
    solver.solve(annotate=annotate)

    return u

u = main(f, annotate=True)
replay_dolfin()

grad = compute_gradient(Functional(u*u*dx*dt[FINISH_TIME]), Control(f),
                        forget = False)

def J(f):
    u = main(f, annotate=False)
    return assemble(u*u*dx)

Ju = assemble(u*u*dx)
minconv = taylor_test(J, Control(f), Ju, grad)
if minconv < 1.9:
    sys.exit(1)
