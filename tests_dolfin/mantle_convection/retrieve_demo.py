from stokes import *
from composition import *
from temperature import *
from parameters import *
from filtering import *
import numpy
from dolfin import *
import time

def store(phi, T, u, t):
    density_series.store(phi.vector(), t)
    temperature_series.store(T.vector(), t)
    velocity_series.store(u.vector(), t)
    if t == 0.0:
        density_series.store(mesh, t)

def message(t, dt):
    print "\n" + "-"*60
    print "t = %0.5g" % t
    print "dt = %0.5g" % dt
    print "-"*60

def preconditioner():
    eta = eta0 * exp(-b_val*T_/deltaT + c_val*(1.0 - triangle.x[1])/height )
    (a, L, pre) = momentum(mesh, eta, g, Ra*T_ - Rb*phi_)
    (A, b) = assemble_system(a, L, bcs)
    P = assemble_system(pre, L, bcs)[0]
    return P

dirname = "res_newD/bin"

velocity_series = TimeSeries("%s/velocity" % dirname)
temperature_series = TimeSeries("%s/temperature" % dirname)
density_series = TimeSeries("%s/density" % dirname)

# Retrieve mesh
mesh = Mesh()
density_series.retrieve(mesh, 0.0)

W = stokes_space(mesh)

# Define function spaces
CG2 = VectorFunctionSpace(mesh, "CG", 2)
DG1 = FunctionSpace(mesh, "DG", 1)

# Define functions for initial data
u_ = Function(CG2)
phi_ = Function(DG1)
T_ = Function(DG1)

# Extract times
times = density_series.vector_times()

# Start at "last time" of store data
t = times[-1]
velocity_series.retrieve(u_.vector(), t)
temperature_series.retrieve(T_.vector(), t)
density_series.retrieve(phi_.vector(), t)

plot(u_)
plot(T_)
plot(phi_)
#interactive()

# Predictor functions
u_pr = Function(CG2)      # Tentative velocity (u)
phi_pr = Function(DG1)    # Tentative composition (phi)
T_pr = Function(DG1)      # Tentative temperature (T)

# Functions at this timestep
u = Function(CG2)         # Velocity (u) at this time step
phi = Function(DG1)       # Composition (phi) at this time step
T = Function(DG1)         # Temperature (T) at this time step

height = 1.0
length = 2.0

# Define initial CLF and time step
maxvel = numpy.max(numpy.abs(u_.vector().array()))
CLFnum = 0.5
norm_u = norm(u)
res = norm_u/sqrt(length*height)
hmin = mesh.hmin()
dt = CLFnum*hmin/maxvel
#CLFpr = 0.0

# Start time loop
t += dt
progress = Progress("Time-stepping")

# Containers for storage of stuff
#dirname = "res_newDTf"

#velocity_series = TimeSeries("%s/bin/velocity" % dirname)
#temperature_series = TimeSeries("%s/bin/temperature" % dirname)
#density_series = TimeSeries("%s/bin/density" % dirname)

# Store initial data
#store(phi_, T_, u_, 0.0)

velocity_pressure = Function(W)

solver = KrylovSolver("tfqmr", "amg_ml")


finish = 1.0

# Define boundary conditions for the composition phi
topR = DirichletBC(DG1, 0.0, "x[1] == 1.0", "geometric")
bottomR = DirichletBC(DG1, 1.0, "x[1] == 0.0", "geometric")
bc = [bottomR]

# Define boundary conditions for the velocity and pressure u
bottom = DirichletBC(W.sub(0), (0.0, 0.0), "x[1] == 0.0" )
top = DirichletBC(W.sub(0).sub(1), 0.0, "x[1] == 1.0" )
left = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == 0.0")
right = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == 2.0")
bcs = [bottom, top, left, right]

# Define boundary conditions for the temperature
top_temperature = DirichletBC(DG1, Constant(0.0), "x[1] == 1.0", "geometric")
bottom_temperature = DirichletBC(DG1, Constant(1.0), "x[1] == 0.0", "geometric")
T_bcs = [bottom_temperature, top_temperature]

P = preconditioner()

while (t <= finish):

    message(t, dt)

    # Solve for predicted phi
    (a, L) = composition(mesh, Constant(dt), u_, phi_)
    eq = VariationalProblem(a, L, bc)
    eq.parameters["solver"]["linear_solver"] = "gmres"
    eq.solve(phi_pr)

    # Filter predicted phi
    filterProp(phi_pr)

    print "Sum of densities =  ", sum(phi_pr.vector().array())

    # Solve for predicted temperature
    (a, L) = temperature(mesh, Constant(dt), u_, T_)
    eq = VariationalProblem(a, L, T_bcs)
    eq.parameters["solver"]["linear_solver"] = "gmres"
    eq.solve(T_pr)

    # Solve for predicted velocity
    H = Ra*T_pr - Rb*phi_pr
    eta = eta0 * exp(-b_val*T_pr/deltaT + c_val*(1.0 - triangle.x[1])/height )
    (a, L, precond) = momentum(mesh, eta, g, H)
    (A, b) = assemble_system(a, L, bcs)
    solver.set_operators(A, P)
    t1 = time.time()
    solver.solve(velocity_pressure.vector(), b)
    t2 = time.time()
    print "Krylov solve took ", (t2 - t1)
    #solve(A, velocity_pressure.vector(), b)
    u_pr.assign(velocity_pressure.split()[0])
    """
    # Solve for corrected phi
    (a, L) = phi_correction(mesh, Constant(dt), u_pr, u_, phi_)
    eq = VariationalProblem(a, L)
    eq.parameters["solver"]["linear_solver"] = "gmres"
    eq.solve(phi)
    """
    phi.assign(phi_pr)

    # Filter corrected phi
    #filterProp(phi)

    print "Sum of corrected densities =  ", sum(phi.vector().array())

    plot(phi, title="density")

    # Solve for corrected temperature T
    (a, L) = temperature_correction(mesh, Constant(dt), u_pr, u_, T_)
    eq = VariationalProblem(a, L, T_bcs)
    eq.parameters["solver"]["linear_solver"] = "gmres"
    eq.solve(T)
    plot(T, title="temperature")


    # Solve for corrected velocity
    H = Ra*T - Rb*phi
    eta = eta0 * exp(-b_val*T/deltaT + c_val*(1.0 - triangle.x[1])/height )

    #solve(A, velocity_pressure.vector(), b)
    (a, L, precond) = momentum(mesh, eta, g, H)
    (A, b) = assemble_system(a, L, bcs)
    solver.set_operators(A, P)
    solver.solve(velocity_pressure.vector(), b)
    u.assign(velocity_pressure.split()[0])

    # Store stuff
    store(phi, T, u, t)

    # Define dt based on CFL condition
    maxvel = numpy.max(numpy.abs(u.vector().array()))
    norm_u = norm(u)
    res = norm_u/sqrt(length*height)
    dt = CLFnum*hmin/maxvel
    print "max vel = ", maxvel
    print "norm_u = ", norm_u
    print "res = ", res
    print

    # Move to new timestep, including updating functions
    phi_.assign(phi)
    T_.assign(T)
    u_.assign(u)

    progress.update(t / finish)
    t += dt

interactive()
