__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2012-02-14

import time
import numpy
import sys
import os
import math

sys.path.insert(0, '..')

from stokes import *
from composition import *
from temperature import *
from parameters import InitialTemperature, Ra, Rb, rho0, g
from parameters import eta0, b_val, c_val, deltaT

from dolfin import *; import dolfin
from dolfin_adjoint import AdjointKrylovMatrix

dolfin.parameters["form_compiler"]["representation"] = "quadrature"

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

def compute_initial_conditions(T_, W, Q, bcs):
    # Solve Stokes problem with given initial temperature and
    # composition
    eta = viscosity(T_)
    (a, L, pre) = momentum(W, eta, (Ra*T_)*g)
    w = Function(W)

    P = PETScMatrix()
    assemble(pre, tensor=P); [bc.apply(P) for bc in bcs]

    solve(a == L, w, bcs=bcs, solver_parameters={"linear_solver": "tfqmr", "preconditioner": "amg"})
    return (w, P)

parameters["form_compiler"]["cpp_optimize"] = True

# Define spatial domain
height = 1.0
length = 2.0
nx = 10
ny = 10
mesh = Rectangle(0, 0, length, height, nx, ny)

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
finish = constant_dt * 10

def main(T_):
  # Define initial and end time
    t = 0.0

    # Define boundary conditions for the velocity and pressure u
    bottom = DirichletBC(W.sub(0), (0.0, 0.0), "x[1] == 0.0" )
    top = DirichletBC(W.sub(0).sub(1), 0.0, "x[1] == %g" % height)
    left = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == 0.0")
    right = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == %g" % length)
    bcs = [bottom, top, left, right]

    rho = interpolate(rho0, Q)

    # Functions at previous timestep (and initial conditions)
    (w_, P) = compute_initial_conditions(T_, W, Q, bcs)

    # Predictor functions
    T_pr = Function(Q)      # Tentative temperature (T)

    # Functions at this timestep
    T = Function(Q)         # Temperature (T) at this time step
    w = Function(W)

    # Define initial CLF and time step
    CLFnum = 0.5
    dt = compute_timestep(w_)
    t += dt
    n = 1

    w_pr = Function(W)
    (u_pr, p_pr) = split(w_pr)
    (u_, p_) = split(w_)

    # Solver for the Stokes systems
    solver = PETScKrylovSolver("gmres", "amg")
    solver.parameters["relative_tolerance"] = 1.0e-14
    solver.parameters["monitor_convergence"] = False

    while (t <= finish):
        message(t, dt)

        # Solve for predicted temperature in terms of previous velocity
        (a, L) = energy(Q, Constant(dt), u_, T_)
        solve(a == L, T_pr, T_bcs, solver_parameters={"krylov_solver": {"relative_tolerance": 1.0e-14}})

        # Solve for predicted flow
        eta = viscosity(T_pr)
        (a, L, precond) = momentum(W, eta, (Ra*T_pr)*g)

        b = assemble(L); [bc.apply(b) for bc in bcs]
        A = AdjointKrylovMatrix(a, bcs=bcs)

        solver.set_operators(A, P)
        solver.solve(w_pr.vector(), b)
        #solve(a == L, w_pr, bcs, solver_parameters={"krylov_solver": {"relative_tolerance": 1.0e-14}}, annotate=annotate)

        # Solve for corrected temperature T in terms of predicted and previous velocity
        (a, L) = energy_correction(Q, Constant(dt), u_pr, u_, T_)
        solve(a == L, T, T_bcs, solver_parameters={"krylov_solver": {"relative_tolerance": 1.0e-14}})

        # Solve for corrected flow
        eta = viscosity(T)
        (a, L, precond) = momentum(W, eta, (Ra*T)*g)

        b = assemble(L); [bc.apply(b) for bc in bcs]
        A = AdjointKrylovMatrix(a, bcs=bcs)

        solver.set_operators(A, P)
        solver.solve(w.vector(), b)
        #solve(a == L, w, bcs, solver_parameters={"krylov_solver": {"relative_tolerance": 1.0e-14}}, annotate=annotate)

        # Compute time step
        dt = compute_timestep(w)

        # Move to new timestep and update functions
        T_.assign(T)
        w_.assign(w)
        t += dt
        n += 1

    return T_

def Nusselt():
    "Definition of Nusselt number, cf Blankenbach et al 1989"

    # Define markers (2) for top boundary, remaining facets are marked
    # by 0
    markers = FacetFunction("uint", mesh)
    markers.set_all(0)
    top = compile_subdomains("near(x[1], %s)" % height)
    top.mark(markers, 2)
    ds = Measure("ds")[markers]

    # Compute \int_bottom T apriori:
    Nu2 = deltaT*length

    return (ds(2), Nu2)

    # Define nusselt number
    #Nu = - (1.0/Nu2)*grad(T)[1]*ds(2)
    #return Nu

if __name__ == "__main__":
    Tic = interpolate(InitialTemperature(Ra, length), Q)
    Tfinal = main(Tic)
