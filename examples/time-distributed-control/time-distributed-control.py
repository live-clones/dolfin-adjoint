#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _klein:
#
# .. py:currentmodule:: dolfin_adjoint
#
# Time-distributed controls
# =========================
#
# .. sectionauthor:: Simon W. Funke <simon@simula.no>
#
#
# Background
# **********
# Some time-dependent problems have control variables that are distributed over
# all (or some) time-levels. The following example demonstrates how this can be
# implemented in dolfin-adjoint.
#
# One important aspect to consider is the regularisation term. For
# time-distributed controls, one typically uses wishes to enforce smoothness
# of the control variables in time. We will also discuss how such a
# regularisation term is implemented.
#
# Problem definition
# ******************
# We consider the heat equation with a time-dependent source term that acts as
# our control variable:
#
# The partial differential equation for this example is the time-dependent heat-equation:
#
# .. math::
#            \frac{\partial u}{\partial t} - \nu \nabla^{2} u= f(t)
#             \quad & \textrm{in\phantom{r} } \Omega \times (0, T), \\
#            u = 0  \quad & \textrm{for } \Omega \times \{0\} \\
#            u = 0  \quad & \textrm{for } \partial \Omega \times (0, T).
#
#
# where :math:`\Omega` is the unit square, :math:`T` is the final time, :math:`u`
# is the unkown temperature variation, :math:`\nu` is the thermal diffusivity, and
# :math:`g` is the initial temperature.
#
# The objective value, the model output of interest, is the norm of the
# temperature variable integrated over time, plus a regularisation term that
# enforces smoothness of the control in time:
#
# .. math::
#            J(u, f) := \int_0^T \int_\Omega u^2 \textrm{d} \Omega \text{d}t +
#                       \frac{\alpha}{2} \int_0^T \int_\Omega \dot f^2 \textrm{d} \Omega \text{d}t
#
# The aim of this example is to compute the sensitivity of :math:`J` with
# respect to the forcing terms :math:`f`, that is:
#
# .. math::
#            \frac{\textrm{d}J}{\textrm{d} f}

# Implementation
# **************

# We start by importing the needed FEniCS and dolfin-adjoint modules (note that
# `fenics_adjoint` is an alias for `dolfin_adjoint`):

from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
dt_meas = dt  # Keep a reference to dt, the time-measure of dolfin-adjoint

# Next, we define the expressions for observational data :math:`d` and the
# viscosity :math:`\nu`.

data = Expression("16*x[0]*(x[0]-1)*x[1]*(x[1]-1)*sin(pi*t)", t=0, degree=4)
nu = Constant(1e-5)

# Next, we define the discretization space:

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "CG", 1)

# ... and time:

dt = Constant(0.25)
T = 1

# We are considering a time-distributed forcing as control. In the next step,
# we create one control function for each timestep in the model, and store all
# controls in a dictionary that maps timestep to control function:

ctrls = OrderedDict()
t = float(dt)
while t <= T:
    ctrls[t] = Function(V, annotate=True)
    t += float(dt)

# The following function implements a heat equation solver in FEniCS. The
# only `dolfin-adjoint` specific functions are `adj_start_timestep` and
# `adj_inc_timestep` to communicute the time-levels to `dolfin_adjoint`, and the
# `annotate` flag in the assignment to enforce that the update of the forcing
# function is captured in the `dolfin-adjoint` tape:

def solve_heat(ctrls):
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V, name="source")
    u_0 = Function(V, name="solution")
    d = Function(V, name="data")

    F = ( (u - u_0)/dt*v + nu*inner(grad(u), grad(v)) - f*v)*dx
    a, L = lhs(F), rhs(F)
    bc = DirichletBC(V, 0, "on_boundary")

    t = float(dt)
    adj_start_timestep(time=t)
    while t <= T:
        # Update source term from control array
        f.assign(ctrls[t])

        # Update data function
        data.t = t
        d.assign(interpolate(data, V), annotate=True)

        # Solve PDE
        solve(a == L, u_0, bc)

        # Update time
        t += float(dt)
        adj_inc_timestep(time=t, finished=t>T)

    return u_0, d

u, d = solve_heat(ctrls)

# With this preparation steps, we are now ready to define the functional.
# First we discretise the regularisation term
#
# .. math::
#             \frac{\alpha}{2} \int_0^T \int_\Omega \dot f^2 \textrm{d} \Omega \text{d}t
#
# Note, that :math:`f` is a piecewise linear function in time over the time intervals :math:`K = [(0, \delta t), (\delta t, 2 \delta t), \dots, (T-\delta
# t, T)]`. Thus, we can write the integral as a sum over all intervals
#
#
# .. math::
#             = \frac{\alpha}{2} \sum_K \int_K \int_\Omega \dot f^2
#             \textrm{d} \Omega\text{d}t
#
# With :math:`K = [a_k, b_k]`, we can continue:
#
# .. math::
#             = \frac{\alpha}{2} \sum_K \int_{a_k}^{b_k} \int_\Omega \dot f(t)^2 \textrm{d} \Omega\text{d}t
#
# Discretising the time-derivative finally yields:
#
# .. math::
#             = \frac{\alpha}{2} \sum_K \int_{a_k}^{b_k}
#             \int_\Omega \left(\frac{f(b_k)-
#             f(a_k)}{b_k-a_k}\right)^2\textrm{d}\Omega \\
#             = \frac{\alpha}{2} \sum_K (b_k-a_k)^{-1}
#             \int_\Omega \left(f(b_k)- f(a_k)\right)^2\textrm{d}\Omega
#
#
# In code this yields:

alpha = Constant(0e-1)
regularisation = alpha/2*sum([1/dt*(fb-fa)**2*dx for fb, fa in
    zip(ctrls.values()[1:], ctrls.values()[:-1])])

# By default, dolfin-adjoint integrates functionals over the full time-interval.
# Since we have manually discretised the functional in time, so it is sufficient
# to let dolfin-adjoint evaluate the functional at the beginning of the
# tape evaluation:

regularisation = regularisation*dt_meas[START_TIME]

# Next, we define the remaining functional terms and controls:

J = Functional((u-d)**2*dx*dt_meas + regularisation)
m = [Control(c) for c in ctrls.values()]

# Finally, we define the reduced functional and solve the optimisation problem:

rf = ReducedFunctional(J, m)
opt_ctrls = minimize(rf, options={"maxiter": 20})

# Depending on the alpha value that we choose, we get different behaviour in the
# controls: the higher the alpha value, the "smoother" the control function will
# be over time.

# .. image:: plot.png
#     :scale: 50
#     :align: center


from matplotlib import pyplot
x = [c.vector().norm("linf") for c in opt_ctrls]
x2 = [norm(c) for c in opt_ctrls]
x3 = [c((0.5, 0.5)) for c in opt_ctrls]
#from IPython import embed; embed()
pyplot.plot(x, label="linf")
pyplot.plot(x2, label="l2")
pyplot.plot(x3, label="midpoint")
pyplot.legend()
pyplot.show()

for i, c in enumerate(opt_ctrls):
    plot(c, title=str(i))
interactive()
