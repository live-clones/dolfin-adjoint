'''Contributed by Martin Alnaes, launchpad question 228870'''

from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

z = Function(R, name="z")

ut = TrialFunction(V)
v = TestFunction(V)
m = Function(V, name="m")
u = Function(V, name="u")

adj_start_timestep(0.0)
nt = 3
for t in range(nt):
    z.interpolate(Expression("t*t", t=t, degree=2), annotate=True) # ... to make sure this is recorded
    solve(ut*v*dx == m*v*dx, u, []) # ... even though it's not used in the computation
    adj_inc_timestep(t+1, finished=t==nt-1)

adj_html("forward.html", "forward")
# ... so it can be replayed by the functional at the end
J = ReducedFunctional(Functional((u-z)**2*dx*dt), Control(m))
assert abs(float(J(m)) - 9.0) < 1.0e-14
