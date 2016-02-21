#!/usr/bin/env python2

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014 University of Edinburgh
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

from exceptions import *

__all__ = \
  [
    "add_parameter",
    "nest_parameters"
  ]

# Enable aggressive compiler optimisations by default.
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

def nest_parameters(parameters, key):
    """
    Create a new Parameters object at the specified key in parameters, if one
    does not already exist.
    """

    if key in parameters:
        if not isinstance(parameters[key], dolfin.Parameters):
            raise ParameterException("Inconsistent parameter type")
    else:
        p = dolfin.Parameters(key)
        parameters.add(p)
    return

def add_parameter(parameters, key, default_value):
    """
    Add a new parameter at the specified key in parameters. If the parameter
    already exists, check that it is of the same type as default_value. Otherwise,
    set the parameter to be equal to default_value.
    """

    if key in parameters:
        if not isinstance(parameters[key], default_value.__class__):
            raise ParameterException("Inconsistent parameter type")
    else:
        parameters.add(key, default_value)
    return

# Configure timestepping parameters.
nest_parameters(dolfin.parameters, "timestepping")
nest_parameters(dolfin.parameters["timestepping"], "pre_assembly")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "linear_forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "bilinear_forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "equations")
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["forms"], "compress_matrices", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["forms"], "expand_form", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["forms"], "term_optimisation", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"], "compress_matrices", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"], "expand_form", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"], "matrix_optimisation", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"], "term_optimisation", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"], "compress_matrices", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"], "expand_form", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"], "term_optimisation", False)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["equations"], "symmetric_boundary_conditions", False)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"], "verbose", True)
