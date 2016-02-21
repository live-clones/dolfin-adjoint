from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = Constant(1)
g = Constant(1)


u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho*u_*v * dx - g*v*dx(domain=mesh)
solve(lhs(F) == rhs(F), u)

J = Functional(0.5 * inner(u, u) * dx + g**3*dx(domain=mesh))

# Reduced functional with single control
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)

assert Jhat.taylor_test(rho, seed=1e-4) > 1.9

direction = Constant(1)
assert Jhat.taylor_test(rho, seed=1e-4, perturbation_direction=direction) > 1.9


# Reduced functional with multiple controls
m2 = Control(g)

Jhat = ReducedFunctional(J, [m, m2])
Jhat.derivative()
Jhat([rho, g])

direction = [Constant(1), Constant(1)]

assert Jhat.taylor_test([rho, g], seed=1e-2) > 1.9
assert Jhat.taylor_test([rho, g], seed=1e-2, perturbation_direction=direction) > 1.9
