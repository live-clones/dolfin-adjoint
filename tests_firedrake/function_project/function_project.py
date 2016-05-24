from firedrake import *
from firedrake_adjoint import *

mesh = UnitIntervalMesh(5)
V = FunctionSpace(mesh, "CG", 1)

c = Function(V, name="Control")
c.project(Constant(1), annotate=False)

u = Function(V, name="Projected")
u.project(c, annotate=True)

J = Functional(u**2*dx)
ctrl = Control(c)
Jhat = ReducedFunctional(J, ctrl)
Jhat.taylor_test(c)
