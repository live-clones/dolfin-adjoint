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
# E. Rognes from dolfin-adjoint file tests/mantle_convection/numerics.py, bzr
# trunk 573
# Code first added: 2013-02-26

# Modified version of mantle_convection test from dolfin-adjoint bzr trunk 513

__license__  = "GNU LGPL Version 3"

from dolfin import *

def advection(phi, psi, u, n, theta=1.0):

    # Define |u * n|
    un = abs(dot(u('+'), n('+')))

    # Contributions from cells
    a_cell = - dot(u*phi, grad(psi))*dx

    # Contributions from interior facets
    a_int = (dot(u('+'), jump(psi, n))*avg(phi)
             + 0.5*un*dot(jump(phi, n), jump(psi, n)))*dS

    return theta*(a_cell + a_int)

def diffusion(phi, psi, k_c, alpha, n, h, theta=1.0):

    # Contribution from the cells
    a_cell = k_c*dot(grad(phi), grad(psi))*dx

    # Contribution from the interior facets
    tmp = (alpha('+')/h('+')*dot(jump(psi, n), jump(phi, n))
           - dot(avg(grad(psi)), jump(phi, n))
           - dot(jump(psi, n), avg(grad(phi))))*dS
    a_int = k_c('+')*tmp

    return theta*(a_cell + a_int)


def backward_euler(u, u_, dt):
    return (u - u_)/dt
