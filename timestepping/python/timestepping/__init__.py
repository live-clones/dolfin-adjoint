#!/usr/bin/env python2

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2016 University of Edinburgh
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

from fenics_patches import *
from parameters import *

import caches
import checkpointing
import equation_solvers
import embedded_cpp
import exceptions
import fenics_overrides
import fenics_utils
import pre_assembled_adjoint
import pre_assembled_equations
import pre_assembled_forms
import statics
import time_functions
import time_levels
import time_systems
import versions
import vtu_io

from caches import *
from checkpointing import *
from equation_solvers import *
from embedded_cpp import *
from exceptions import *
from fenics_overrides import *
from fenics_utils import *
from pre_assembled_adjoint import *
from pre_assembled_equations import *
from pre_assembled_forms import *
from statics import *
from time_functions import *
from time_levels import *
from time_systems import *
from versions import *
from vtu_io import *

__doc__ = \
"""
A timestepping abstraction and automated adjoining library. This library
utilises the FEniCS system for symbolic manipulation and automated code
generation, and supplements this system with a syntax for the description of
timestepping finite element models.
"""

__license__ = "LGPL-3"

__version__ = "1.5.0+"

__all__ = \
  caches.__all__ + \
  checkpointing.__all__ + \
  equation_solvers.__all__ + \
  embedded_cpp.__all__ + \
  exceptions.__all__ + \
  fenics_overrides.__all__ + \
  fenics_patches.__all__ + \
  fenics_utils.__all__ + \
  parameters.__all__ + \
  pre_assembled_adjoint.__all__ + \
  pre_assembled_equations.__all__ + \
  pre_assembled_forms.__all__ + \
  statics.__all__ + \
  time_functions.__all__ + \
  time_levels.__all__ + \
  time_systems.__all__ + \
  versions.__all__ + \
  vtu_io.__all__ + \
  [
    "caches",
    "checkpointing",
    "equation_solvers",
    "embedded_cpp",
    "exceptions",
    "fenics_overrides",
    "fenics_utils",
    "pre_assembled_adjoint",
    "pre_assembled_equations",
    "pre_assembled_forms",
    "statics",
    "time_functions",
    "time_levels",
    "time_systems",
    "versions",
    "vtu_io"
  ]
