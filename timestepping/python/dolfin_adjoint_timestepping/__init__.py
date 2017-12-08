#!/usr/bin/env python3

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2017 University of Edinburgh
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

import dolfin_adjoint
from . import dolfin_adjoint_timestepping

import timestepping

from timestepping import *
from .dolfin_adjoint_timestepping import *

__doc__ = \
"""
A timestepping abstraction and automated adjoining library. This library
utilises the FEniCS system for symbolic manipulation and automated code
generation, and supplements this system with a syntax for the description of
timestepping finite element models.

This version of the library integrates with dolfin-adjoint.
"""

__license__ = "LGPL-3"

__version__ = "1.6.0+"

__all__ = \
  dolfin_adjoint_timestepping.__all__ + \
  [
    "dolfin_adjoint",
    "dolfin_adjoint_timestepping",
    "timestepping"
  ]
