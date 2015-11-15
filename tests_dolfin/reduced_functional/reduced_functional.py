from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = interpolate(Constant(1), W, name="Control")
g = interpolate(Constant(1), W, name="Control2")


u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho * u_ * v * dx - g*v*dx
solve(lhs(F) == rhs(F), u)

J = Functional(0.5 * inner(u, u) * dx + g**2*dx)
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)
Jhat.hessian(rho)

direction = interpolate(Constant(1), W)
Jhat.taylor_test(rho, perturbation_direction=direction)


m2 = Control(g)

Jhat = ReducedFunctional(J, [m, m2])
Jhat.derivative()
Jhat([rho, g])
Jhat.hessian([rho, g])

direction = [interpolate(Constant(1), W), interpolate(Constant(1), W)]
Jhat.taylor_test([rho, g], perturbation_direction=direction)
