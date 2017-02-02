from firedrake import *

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
mass = inner(u, v)*dx
M = assemble(mass)
data = Function(V)
data.vector()[0] = 1.0
rhs = M*data.vector()
