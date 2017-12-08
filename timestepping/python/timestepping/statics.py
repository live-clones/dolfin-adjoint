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
    "StaticConstant",
    "StaticDirichletBC",
    "StaticFunction",
    "extract_non_static_coefficients",
    "is_static_coefficient",
    "is_static_bc",
    "is_static_form",
    "n_non_static_bcs",
    "n_non_static_coefficients"
  ]

def StaticConstant(*args, **kwargs):
    """
    Return a Constant which is marked as "static". Arguments are identical to the
    Constant function.
    """

    c = dolfin.Constant(*args, **kwargs)
    if isinstance(c, ufl.tensors.ListTensor):
        for c_c in c:
            assert(isinstance(c_c, dolfin.Constant))
            c_c._time_static = True
    else:
        assert(isinstance(c, dolfin.Constant))
        c._time_static = True

    return c

def StaticFunction(*args, **kwargs):
    """
    Return a Function which is marked as "static". Arguments are identical to the
    Function function.
    """

    fn = dolfin.Function(*args, **kwargs)
    fn._time_static = True

    return fn

class StaticDirichletBC(DirichletBC):
    """
    A DirichletBC which is marked as "static". Constructor arguments are identical
    to the DOLFIN DirichletBC constructor.
    """

    def __init__(self, *args, **kwargs):
        DirichletBC.__init__(self, *args, **kwargs)

        self._time_static = True

        return

def is_static_coefficient(c):
    """
    Return whether the supplied argument is a static Coefficient.
    """

    return isinstance(c, ufl.constantvalue.ConstantValue) or (hasattr(c, "_time_static") and c._time_static)

def extract_non_static_coefficients(form):
    """
    Return all non-static Coefficient s associated with the supplied form.
    """

    non_static = []
    for c in ufl.algorithms.extract_coefficients(form):
        if not is_static_coefficient(c):
            non_static.append(c)
    return non_static

def n_non_static_coefficients(form):
    """
    Return the number of non-static Coefficient s associated with the supplied
    form.
    """

    non_static = 0
    for c in ufl.algorithms.extract_coefficients(form):
        if not is_static_coefficient(c):
            non_static += 1
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
    Return whether the supplied DirichletBC is "static".
    """

    if not isinstance(bc, dolfin.cpp.DirichletBC):
        raise InvalidArgumentException("bc must be a DirichletBC")

    return hasattr(bc, "_time_static") and bc._time_static

def n_non_static_bcs(bcs):
    """
    Given a list of DirichletBC s, return the number of static DirichletBC s.
    """

    if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    n = 0
    for bc in bcs:
        if not is_static_bc(bc):
            n += 1

    return n
