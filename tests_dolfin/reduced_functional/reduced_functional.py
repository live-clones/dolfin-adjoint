from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = interpolate(Constant(1), W, name="Control")
g = interpolate(Constant(1), W, name="Control2")


u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho*u_*v * dx - Constant(1)*g*v*dx
solve(lhs(F) == rhs(F), u)

J = Functional(0.5 * inner(u, u) * dx + g**3*dx)

# Reduced functional with single control
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)
Jhat.hessian(rho)

direction = interpolate(Constant(1), W)
assert Jhat.taylor_test(rho, test_hessian=True, perturbation_direction=direction) > 2.9


# Reduced functional with multiple controls
m2 = Control(g)

Jhat = ReducedFunctional(J, [m, m2])
Jhat.derivative()
Jhat([rho, g])
Jhat.hessian([rho, g])

direction = [interpolate(Constant(1), W), interpolate(Constant(10), W)]
assert Jhat.taylor_test([rho, g], test_hessian=True, perturbation_direction=direction) > 2.9
assert Jhat.taylor_test([rho, g]) > 1.9
