

from numpy import *
from time import sleep
from dolfin import *

dirname = "bin-final"
velocities = TimeSeries("%s/velocity" % dirname)
temperatures = TimeSeries("%s/temperature" % dirname)
densities = TimeSeries("%s/density" % dirname)

height = 1.0
deltaT = 1.0
b_val = ln(2.5)
c_val = ln(2.0)
eta0 = 1.0


# Retrieve mesh
mesh = Mesh()
densities.retrieve(mesh, 0.0)

CG2 = VectorFunctionSpace(mesh, "CG", 2)
DG1 = FunctionSpace(mesh, "DG", 1)

velocity = Function(CG2)
density = Function(DG1)
temperature = Function(DG1)
viscosity = Function(DG1)

vfile = File("%s/pvd/velocity.pvd" % dirname)
tfile = File("%s/pvd/temperature.pvd" % dirname)
rfile = File("%s/pvd/density.pvd" % dirname)
vsfile = File("%s/pvd/viscosity.pvd" % dirname)


eta = eta0 * exp(-abs(b_val*temperature)/deltaT + c_val*(1.0 - triangle.x[1])/height )

times = densities.vector_times()

for t in times:

    # Retrieve data at this time step
    densities.retrieve(density.vector(), t)
    velocities.retrieve(velocity.vector(), t)
    temperatures.retrieve(temperature.vector(), t)

    tmp = project(eta, DG1)
    viscosity.assign(tmp)

    #plot(density, title="Density")
    #plot(viscosity, title = "Viscosity")
    #plot(velocity, title="Velocity")
    #plot(temperature, title="Temperature")

    # Output to vtk
    if True:
        vfile << velocity
        tfile << temperature
        rfile << density
        vsfile << viscosity

#interactive()
