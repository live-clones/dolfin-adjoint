from dolfin import *
from dolfin_adjoint import *
import numpy as np
import os, sys

# Set log level
set_log_level(WARNING)

# Prepare a mesh
mesh = UnitIntervalMesh(50)

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
    def __init__(self, omega=Constant(2e2), derivative=None, degree=3):
        """ Construct the source function """
        self.t = 0.0
        self.omega = omega
        self.derivative = derivative
        
    def eval(self, value, x):
        """ Evaluate the expression """
        if self.derivative is None:        
            if x[0] < 1e-15:
                value[0] = np.sin(float(self.omega)*self.t)
            else:
                value[0] = 0.
        elif self.derivative == self.omega:
            if x[0] < 1e-15:
                value[0] = self.t*np.cos(float(self.omega)*self.t)
            else:
                value[0] = 0.

def forward(excitation, c=Constant(1.), record=False, annotate=False):
    # Define function space
    U = FunctionSpace(mesh, "Lagrange", 1)

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
    u = Function(U, name = "u", annotate = annotate)

    # The actual timestepping
    if record: rec = [u1(1.),]
    i = 1
    t = 0.0        # Initial time
    T = 3.e-1      # Final time
    times = [t,]
    if annotate: adj_start_timestep()
    while t < T - .5*float(k):
        excitation.t = t + float(k)
        solve(a == L, u, annotate = annotate)
        u0.assign(u1, annotate = annotate)
        u1.assign(u, annotate = annotate)

        t = i*float(k)
        times.append(t)
        if record:
            rec.append(u1(1.0))
        if annotate: adj_inc_timestep(t, t > T - .5*float(k))
        i += 1

    if record:
        np.savetxt("recorded.txt", rec)

    return u1, times

# Callback function for the optimizer
# Writes intermediate results to a logfile
def eval_cb(j, m):
    print("omega = %15.10e " % float(m[0]))
    print("objective = %15.10e " % j)

# Prepare the objective function
def objective(times, u, observations):
    combined = zip(times, observations)
    area = times[-1] - times[0]
    M = len(times)
    I = area/M*sum(inner(u - u_obs, u - u_obs)*ds(1)*dt[t]
                   for (t, u_obs) in combined)
    return I

def optimize(dbg=False):
    # Define the control
    Omega = Constant(190)
    source = Source(Omega)
    source.dependencies = Omega  # dolfin-adjoint needs to know on which
                                 # coefficients this expression depends on
    # Provide the derivative coefficients
    source.user_defined_derivatives = {Omega: Source(Omega, derivative=Omega)}

    # Execute first time to annotate and record the tape
    u, times = forward(source, 2*DOLFIN_PI, False, True)
    print "recording completed"
    if dbg:
        # Check the recorded tape
        success = replay_dolfin(tol = 0.0, stop = True)
        print "replay: ", success

        # for the equations recorded on the forward run
        adj_html("forward.html", "forward")
        # for the equations to be assembled on the adjoint run
        adj_html("adjoint.html", "adjoint")

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

    print "define controls"
    # Define the controls
    controls = Control(Omega)

    print "define objective"
    Jform = objective(times, u, refs)
    print "define functional"    
    J = Functional(Jform)
    print "compute gradient"
    # compute the gradient
    dJd0 = compute_gradient(J, controls)
    print float(dJd0[0])

    # Prepare the reduced functional
    reduced_functional = ReducedFunctional(J, controls, eval_cb_post = eval_cb)

    print "optimize"
    # Run the optimisation
    omega_opt = minimize(reduced_functional, method = "L-BFGS-B",\
                     tol=1.0e-12, options = {"disp": True,"gtol":1.0e-12})

    # Print the obtained optimal value for the controls
    print "omega = %f" %float(omega_opt)

if __name__ == "__main__":
    if '-r' in sys.argv:
        print "compute reference solution"
        os.popen('rm -rf recorded.txt')
        source = Source(Constant(2e2))
        forward(source, 2*DOLFIN_PI, True)
    print "start automatic characterization"
    if '-dbg' in sys.argv:
        optimize(True)
    else:
        optimize()
