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
# E. Rognes from dolfin-adjoint file tests/mantle_convection/stokes.py, bzr
# trunk 573
# Code first added: 2013-02-26

# Modified version of mantle_convection test from dolfin-adjoint bzr trunk 513

__license__  = "GNU LGPL Version 3"

from dolfin import *

def stokes_space(mesh):
    # Define spatial discretization (Taylor--Hood)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    return W

def strain(v):
    return sym(grad(v))

def momentum(W, eta, f):

    # Define basis functions
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Define equation F((u, p), (v, q)) = 0
    F = (2.0*eta*inner(strain(u), strain(v))*dx
         + div(v)*p*dx
         + div(u)*q*dx
         + inner(f, v)*dx)

    # Define form for preconditioner
    precond = inner(grad(v), grad(u))*dx + q*p*dx

    # Return left and right hand side and preconditioner
    return (lhs(F), rhs(F), precond)
