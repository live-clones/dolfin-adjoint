""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

    subjecct to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""
from dolfin import *
from dolfin_adjoint import *


set_log_level(ERROR)

# Create mesh, refined in the center
n = 64
mesh = UnitSquareMesh(n, n)

#cf = CellFunction("bool", mesh)
#subdomain = CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
#subdomain.mark(cf, True)
#mesh = refine(mesh, cf)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("0.0"), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile

alpha = Constant(1e-6)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)
f_opt = minimize(rf, bounds=(0.0, 0.8), tol=1.0e-10, options={"gtol": 1.0e-10, "factr": 0.0})
plot(f_opt, interactive=True)
