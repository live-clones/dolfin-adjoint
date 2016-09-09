from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4, 4)
V0 = FunctionSpace(mesh, "CG", 1)

u0 = interpolate(Constant(1), V0, name="InitialCondition", annotate=False)
u0.assign(10*u0, annotate=False)

assert adjointer.adjointer.equations_sz == 0
