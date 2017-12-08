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
# E. Rognes from dolfin-adjoint file tests/mantle_convection/composition.py, bzr
# trunk 573
# Code first added: 2013-02-26

# Modified version of mantle_convection test from dolfin-adjoint bzr trunk 513

__license__  = "GNU LGPL Version 3"

from dolfin import *
from timestepping import *
from numerics import advection, diffusion, backward_euler

def transport(Q, dt, u, phi_):

    # Define test and trial functions
    phi = TrialFunction(Q)
    v = TestFunction(Q)

    # Constants associated with DG scheme
    alpha = StaticConstant(10.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Diffusivity constant
    kappa = StaticConstant(0.0001)

    # Define discrete time derivative operator
    Dt =  lambda phi:    backward_euler(phi, phi_, dt)
    a_A = lambda phi, v: advection(phi, v, u, n)
    a_D = lambda phi, v: diffusion(phi, v, kappa, alpha, n, h)

    # Define form
    F = Dt(phi)*v*dx + a_A(phi, v) + a_D(phi, v)

    return (lhs(F), rhs(F))
