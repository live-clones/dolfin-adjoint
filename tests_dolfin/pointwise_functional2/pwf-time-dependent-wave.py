from dolfin import *
from dolfin_adjoint import *

from IPython import embed as key

import numpy as np
import os, sys

# Set log level
set_log_level(WARNING)

# Prepare a mesh
mesh = UnitIntervalMesh(50)

# Define function space
U = FunctionSpace(mesh, "Lagrange", 1)

# Choose a time step size
k = Constant(1e-3)

# Compile sub domains for boundaries
left  = CompiledSubDomain("near(x[0], 0.)")
right = CompiledSubDomain("near(x[0], 1.)")

# Label boundaries, required for the objective
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
left.mark(boundary_parts, 0)
right.mark(boundary_parts, 1)
ds = Measure("ds")[boundary_parts]

class Source(Expression):
    def __init__(self, t, omega=Constant(2e2)):
        """ Construct the source function """
        self.t = t
        self.omega = omega

    def eval(self, value, x):
        """ Evaluate the expression """
        if x[0] < 1e-15:
            value[0] = np.sin(float(self.omega)*self.t)
        else:
            value[0] = 0.

def forward(c=Constant(1.), record=False, annotate=False):
    excitation = Source(t = 0.0, omega = Constant(190))
    # Set up initial values
    u0 = interpolate(Expression("0."), U, name = "u0", annotate = annotate)
    u1 = interpolate(Expression("0."), U, name = "u1", annotate = annotate)

    # Define test and trial functions
    v = TestFunction(U)
    u = TrialFunction(U)

    # Define variational formulation
    udot = (u - 2.*u1 + u0)
    uold = (0.25*u + 0.5*u1 +0.25*u0)
    F = (udot*v+k*k*c*c*uold.dx(0)*v.dx(0))*dx - u*v*ds(0) + excitation*v*ds(0)
    a = lhs(F)
    L = rhs(F)

    # Prepare solution
    u = Function(U, name = "u", annotate=annotate)

    # The actual timestepping
    if record: rec = [u1(1.),]
    i = 1
    t = 0.0        # Initial time
    T = 3.e-1      # Final time
    times = [t,]
    if annotate: adj_start_timestep()
    while t < T - .5*float(k):
        print t
        excitation.t = t + float(k)
        solve(a == L, u, annotate=annotate)
        u0.assign(u1, annotate=annotate)
        u1.assign(u, annotate=annotate)

        t = i*float(k)
        times.append(t)
        if record:
            rec.append(u1(1.0))
            plot(u)
        if annotate: adj_inc_timestep(t, t > T - .5*float(k))
        i += 1

    if record:
        np.savetxt("recorded.txt", rec)

    return u1, times

# Callback function for the optimizer
# Writes intermediate results to a logfile
def eval_cb(j, m):
    plot(m)
    print("objective = %15.10e " % j)

# Prepare the objective function
def objective(times, u, observations):
    combined = zip(times, observations)
    I = sum(inner(u - u_obs, u - u_obs)*ds(1)*dt[t]
                   for (t, u_obs) in combined)
    return I

def optimize(dbg=False):
    # Execute first time to annotate and record the tape
    c = Function(U, name="control")
    c.assign(interpolate(Constant(2*pi), U))
#    c = Constant(2*pi, name="control")
    u, times = forward(c, False, True)

    if dbg:
        # Check the recorded tape
        success = replay_dolfin(tol = 0.0, stop = True)
        print "replay: ", success

        adj_html("forward.html", "forward") # for the equations recorded on the forward run
        adj_html("adjoint.html", "adjoint") # for the equations to be assembled on the adjoint run

    # Load references
    refs = np.loadtxt("recorded.txt")

    # create noise to references
    gamma = 1.e-5
    if gamma > 0:
        noise = np.random.normal(0, gamma, refs.shape[0])

        # add noise to the refs
        refs += noise

    # map refs to be constant
    refs = map(Constant, refs)

    # Define the controls
    controls = Control(c)

    Jform = PointwiseFunctional(u, refs, np.array([1.0]), times)
    J = Functional(Jform)

    # compute the gradient
    dJd0 = compute_gradient(J, controls, forget=False)
    #plot(dJd0, interactive=True)

    # Prepare the reduced functional
    reduced_functional = ReducedFunctional(J, controls, eval_cb_post = eval_cb)

    # Run the optimisation
    omega_opt = minimize(reduced_functional, method = "L-BFGS-B",\
                     tol=1.0e-12, options = {"disp": True,"gtol":1.0e-12})

    # Print the obtained optimal value for the controls
    print "omega = %f" %float(omega_opt)

if __name__ == "__main__":
    if '-r' in sys.argv:
        print "compute reference solution"
        os.popen('rm -rf recorded.txt')
        forward(Constant(2*DOLFIN_PI), True)
    print "start automatic characterization"
    if '-dbg' in sys.argv:
        optimize(True)
    else:
        optimize()
