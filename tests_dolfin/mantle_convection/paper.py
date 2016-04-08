__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2013-06-14

import time
import numpy
import sys
import os
import math

from stokes import *
from composition import *
from temperature import *
from parameters import InitialTemperature, Ra, Rb, rho0, g
from parameters import eta0, b_val, c_val, deltaT

from dolfin import *; import dolfin
from dolfin_adjoint import *
dolfin.parameters["adjoint"]["fussy_replay"] = True
#dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["form_compiler"]["representation"] = "quadrature"
dolfin.parameters["num_threads"] = 8

def viscosity(T):
    eta = eta0 * exp(-b_val*T/deltaT + c_val*(1.0 - triangle.x[1])/height )
    return eta

def store(T, w, t):
    temperature_series << (T, t)
    flow_series << (w, t)

def message(t, dt):
    print "\n" + "-"*60
    print "t = %0.5g" % t
    print "dt = %0.5g" % dt
    os.system("date")
    print "-"*60

def compute_timestep(w):
  #(u, p) = w.split(deepcopy=True)
  #  maxvel = numpy.max(numpy.abs(u.vector().array()))
  #  mesh = u.function_space().mesh()
  #  hmin = mesh.hmin()
  #  dt = CLFnum*hmin/maxvel

    dt = constant_dt
    return dt

def compute_initial_conditions(T_, W, Q, bcs, annotate):
    # Solve Stokes problem with given initial temperature and
    # composition
    eta = viscosity(T_)
    (a, L, pre) = momentum(W, eta, (Ra*T_)*g)
    w = Function(W)

    P = PETScMatrix()
    assemble(pre, tensor=P); [bc.apply(P) for bc in bcs]

    solve(a == L, w, bcs=bcs, solver_parameters={"linear_solver": "default", "preconditioner": "default"}, annotate=annotate)
    return (w, P)

parameters["form_compiler"]["cpp_optimize"] = True

# Define spatial domain
height = 1.0
length = 2.0
nx = 40
ny = 40
mesh = RectangleMesh(0, 0, length, height, nx, ny)

# Containers for storage
flow_series = File("bin-final/flow.pvd", "compressed")
temperature_series = File("bin-final/temperature.pvd", "compressed")

# Create function spaces
W = stokes_space(mesh)
V = W.sub(0).collapse()
Q = FunctionSpace(mesh, "DG", 1)

print "Number of degrees of freedom:", (W*Q).dim()

# Define boundary conditions for the temperature
top_temperature = DirichletBC(Q, 0.0, "x[1] == %g" % height, "geometric")
bottom_temperature = DirichletBC(Q, 1.0, "x[1] == 0.0", "geometric")
T_bcs = [bottom_temperature, top_temperature]
constant_dt = 3.0e-5
finish = 0.005
#finish = 10*constant_dt

adj_checkpointing('multistage', steps=int(math.floor(finish/constant_dt)), snaps_on_disk=30, snaps_in_ram=30, verbose=True)

def main(T_ic, annotate=False):
  # Define initial and end time
    t = 0.0

    T_ = Function(T_ic, annotate=annotate)

    # Define boundary conditions for the velocity and pressure u
    bottom = DirichletBC(W.sub(0), (0.0, 0.0), "x[1] == 0.0" )
    top = DirichletBC(W.sub(0).sub(1), 0.0, "x[1] == %g" % height)
    left = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == 0.0")
    right = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == %g" % length)
    evil = DirichletBC(W.sub(1), 0.0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS",
                       "pointwise")
    bcs = [bottom, top, left, right, evil]

    rho = interpolate(rho0, Q)

    # Functions at previous timestep (and initial conditions)
    (w_, P) = compute_initial_conditions(T_, W, Q, bcs, annotate=annotate)

    # Predictor functions
    T_pr = Function(Q)      # Tentative temperature (T)

    # Functions at this timestep
    T = Function(Q)         # Temperature (T) at this time step
    w = Function(W)

    # Store initial data
    if annotate:
        store(T_, w_, 0.0)

    # Define initial CLF and time step
    CLFnum = 0.5
    dt = compute_timestep(w_)
    t += dt
    n = 1

    w_pr = Function(W)
    (u_pr, p_pr) = split(w_pr)
    (u_, p_) = split(w_)

    while (t <= finish):
        message(t, dt)

        # Solve for predicted temperature in terms of previous velocity
        (a, L) = energy(Q, Constant(dt), u_, T_)
        solve(a == L, T_pr, T_bcs, annotate=annotate)

        # Solve for predicted flow
        eta = viscosity(T_pr)
        (a, L, precond) = momentum(W, eta, (Ra*T_pr)*g)

        solve(a == L, w_pr, bcs, annotate=annotate)

        # Solve for corrected temperature T in terms of predicted and
        # previous velocity
        (a, L) = energy_correction(Q, Constant(dt), u_pr, u_, T_)
        solve(a == L, T, T_bcs, annotate=annotate)

        # Solve for corrected flow
        eta = viscosity(T)
        (a, L, precond) = momentum(W, eta, (Ra*T)*g)
        solve(a == L, w, bcs, annotate=annotate)

        # Store stuff
        if annotate:
            store(T, w, t)

        # Compute time step
        dt = compute_timestep(w)

        # Move to new timestep and update functions
        T_.assign(T)
        w_.assign(w)
        t += dt
        n += 1
        adj_inc_timestep()

    return T_

def Nusselt():
    "Definition of Nusselt number, cf Blankenbach et al 1989"

    # Define markers (2) for top boundary, remaining facets are marked
    # by 0
    markers = FacetFunction("size_t", mesh)
    markers.set_all(0)
    top = compile_subdomains("near(x[1], %s)" % height)
    top.mark(markers, 2)
    ds = Measure("ds")[markers]

    # Compute \int_bottom T apriori:
    Nu2 = deltaT*length

    return (ds(2), Nu2)

if __name__ == "__main__":
    Tic = interpolate(InitialTemperature(Ra, length), Q, name="InitialTemperature")

    Tfinal = main(Tic, annotate=True)
    (ds2, Nu2) = Nusselt()

    J = Functional(-(1.0/Nu2)*grad(Tfinal)[1]*ds2*dt[FINISH_TIME])
    m = InitialConditionParameter("InitialTemperature")
    Jm = assemble(-(1.0/Nu2)*grad(Tfinal)[1]*ds2)
    dJdm = compute_gradient(J, m, forget=False)

    # project to trial space
    M = assemble(inner(TestFunction(Q), TrialFunction(Q))*dx)
    dJdm_p = Function(Q)
    solve(M, dJdm_p.vector(), dJdm.vector())

    adjoint_vtu = File("bin-final/gradient.pvd", "compressed")
    adjoint_vtu << dJdm_p

    def J(ic):
        Tfinal = main(ic)
        return assemble(-(1.0/Nu2)*grad(Tfinal)[1]*ds2)

    perturbation_direction = Function(Q)
    perturbation_direction.vector()[:] = 1.0
    minconv = taylor_test(J, m, Jm, dJdm)
