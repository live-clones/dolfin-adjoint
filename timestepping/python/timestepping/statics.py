#!/usr/bin/env python3

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

import dolfin
import ufl

from .exceptions import *
from .fenics_overrides import *

__all__ = \
  [
    "Static",
    "StaticConstant",
    "StaticDirichletBC",
    "StaticFunction",
    "extract_non_static_coefficients",
    "is_static_coefficient",
    "is_static_bc",
    "is_static_bcs",
    "is_static_form"
  ]

class Static:
    """
    Used to mark objects as "static".
    """
    pass

class StaticConstant(dolfin.Constant, Static):
    """
    A Constant which is marked as "static".
    """
    pass

class StaticFunction(dolfin.Function, Static):
    """
    A Function which is marked as "static".
    """
    pass

class StaticDirichletBC(dolfin.DirichletBC, Static):
    """
    A DirichletBC which is marked as "static".
    """
    pass

def is_static_coefficient(c):
    """
    Return whether the supplied argument is a static Coefficient.
    """

    return isinstance(c, (ufl.constantvalue.ConstantValue, Static))

def extract_non_static_coefficients(form):
    """
    Return all non-static Coefficient s associated with the supplied form.
    """

    non_static = []
    for c in ufl.algorithms.extract_coefficients(form):
        if not is_static_coefficient(c):
            non_static.append(c)
    return non_static

def is_static_form(form):
    """
    Return whether the supplied form is "static".
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    for dep in ufl.algorithms.extract_coefficients(form):
        if not is_static_coefficient(dep):
            return False
    return True

def is_static_bc(bc):
    """
    Return whether the supplied argument is a static DirichletBC.
    """
    
    return isinstance(bc, StaticDirichletBC)

def is_static_bcs(bcs):
    """
    Return whether the supplied list of DirichletBC s contains only
    StaticDirichletBC s.
    """

    if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
        if not isinstance(bc, dolfin.cpp.DirichletBC):
            raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    for bc in bcs:
        if not is_static_bc(bc):
            return False
    return True
