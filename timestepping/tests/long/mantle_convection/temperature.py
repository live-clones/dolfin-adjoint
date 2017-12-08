#!/usr/bin/env python3

# Copyright (C) 2011 Simula Research Laboratory and Lyudmyla Vynnytska and Marie
#                    E. Rognes
# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014, 2017 University of Edinburgh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright (C) 2011 Simula Research Laboratory and Lyudmyla Vynnytska and Marie
# E. Rognes from dolfin-adjoint file tests/mantle_convection/temperature.py, bzr
# trunk 573
# Code first added: 2013-02-26

# Modified version of mantle_convection test from dolfin-adjoint bzr trunk 513

__license__  = "GNU LGPL Version 3"

from dolfin import *
from timestepping import *
from numerics import advection, diffusion, backward_euler

def energy(Q, dt, u, T_):

    # Define basis functions
    T = TrialFunction(Q)
    psi = TestFunction(Q)

    # Diffusivity constant
    kappa = StaticConstant(1.0)

    # Variables associated with DG scheme
    alpha = StaticConstant(50.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Define discrete time derivative operator
    Dt =  lambda T:    backward_euler(T, T_, dt)
    a_A = lambda u, T, psi: advection(T, psi, u, n)
    a_D = lambda T, psi: diffusion(T, psi, kappa, alpha, n, h)

    # Define form
    F = Dt(T)*psi*dx + a_A(u, T, psi) + a_D(T, psi)

    return (lhs(F), rhs(F))

def energy_correction(Q, dt, u, u_, T_):

    # Define test and trial functions
    T = TrialFunction(Q)
    psi = TestFunction(Q)

    # Diffusivity constant
    k_c = StaticConstant(1.0)

    # Constants associated with DG scheme
    alpha = StaticConstant(50.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Define discrete time derivative operator
    def Dt(T):
        return backward_euler(T, T_, dt)

    # Add syntactical sugar for a_A and a_D
    def a_A(u, T, psi):
        return advection(T, psi, u, n, theta=0.5)
    def a_D(T, psi):
        return diffusion(T, psi, k_c, alpha, n, h, theta=0.5)

    # Define form
    F = (Dt(T)*psi*dx
         + a_A(u, T, psi) + a_A(u_, T_, psi)
         + a_D(T, psi) + a_D(T_, psi))

    return (lhs(F), rhs(F))
