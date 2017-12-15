#!/usr/bin/env python3

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2017 The University of Edinburgh
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

from collections import OrderedDict
import copy

import dolfin
import ufl

from .exceptions import *
from .fenics_overrides import *
from .fenics_utils import *
from .versions import *

__all__ = \
  [
    "AssemblyCache",
    "SolverCache",
    "assembly_cache",
    "linear_solver_cache",
    "cache_info",
    "clear_caches"
  ]

def cache_info(msg, info = dolfin.info):
    """
    Print a message if verbose pre-assembly is enabled.
    """

    if dolfin.parameters["timestepping"]["pre_assembly"]["verbose"]:
        info(msg)
    return

def form_key(form, static = True):
    """
    Generate a hashable key from a Form.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    if static:
        return expand(form)
    else:
        return extract_test_and_trial(form)

def bc_key(bcs, symmetric_bcs, zero_columns = True):
    """
    Generate a hashable key from a list of DirichletBC s.
    """

    if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
        if not isinstance(bc, dolfin.cpp.DirichletBC):
            raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    if len(bcs) == 0:
        return None
    else:
        return tuple(bcs), symmetric_bcs, zero_columns

def parameters_key(parameters):
    """
    Generate a hashable key from a Parameters.
    """

    if not isinstance(parameters, (dolfin.Parameters, dict)):
        raise InvalidArgumentException("parameters must be a Parameters or dictionary")

    fparameters = []
    for key in sorted(parameters.keys()):
        if isinstance(parameters[key], (dolfin.Parameters, dict)):
            fparameters.append(parameters_key(parameters[key]))
        else:
            fparameters.append((key, parameters[key]))
    return tuple(fparameters)

class AssemblyCache:
    """
    A cache of assembled Form s. The assemble method can be used to assemble a
    given Form. If an assembled version of the Form exists in the cache, then the
    cached result is returned. Note that this does not check that the Form
    dependencies are unchanged between subsequent assemble calls -- that is
    deemed the responsibility of the caller.
    """

    def __init__(self):
        self.__cache = {}

        return

    def assemble(self, form, form_compiler_parameters = {}, bcs = [],
      symmetric_bcs = False, zero_columns = True):
        """
        Return the result of assembling the supplied Form.

        Arguments:
          form: The form.
          form_compiler_parameters: Form compiler parameters.
          bcs: Dirichlet BCs applied to a matrix.
          symmetric_bcs: Whether Dirichlet BCs should be applied so as to yield a
            symmetric matrix.
          zero_columns: Whether the zero_columns method should be used to
            apply boundary conditions. If False then the boundary conditions
            should be homogeneous.
        """

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")
        if not isinstance(form_compiler_parameters, (dolfin.Parameters, dict)):
            raise InvalidArgumentException("form_compiler_parameters must be a Parameters or dictionary")
        if not isinstance(bcs, list):
            raise InvalidArgumentException("bcs must be a list of DirichletBC s")
        for bc in bcs:
            if not isinstance(bc, dolfin.cpp.DirichletBC):
                raise InvalidArgumentException("bcs must be a list of DirichletBC s")

        nform_compiler_parameters = dolfin.parameters["form_compiler"].copy()
        nform_compiler_parameters.update(form_compiler_parameters)
        form_compiler_parameters = nform_compiler_parameters;  del(nform_compiler_parameters)

        rank = form_rank(form)
        key = (form_key(form), parameters_key(form_compiler_parameters), bc_key(bcs, symmetric_bcs, zero_columns = zero_columns))
        if len(bcs) == 0:
            if not key in self.__cache:
                cache_info("Assembling form with rank %i" % rank, dolfin.info)
                self.__cache[key] = assemble(form, form_compiler_parameters = form_compiler_parameters)
            else:
                cache_info("Using cached assembled form with rank %i" % rank, dolfin.info)
        else:
            if not rank == 2:
                raise InvalidArgumentException("form must be rank 2 when applying boundary conditions")

            if not key in self.__cache:
                cache_info("Assembling form with rank 2, with boundary conditions", dolfin.info)
                if symmetric_bcs and not zero_columns:
                    mat = assemble_symmetric_bcs(form, bcs, form_compiler_parameters = form_compiler_parameters)
                else:
                    mat = assemble(form, form_compiler_parameters = form_compiler_parameters)
                    apply_bcs(mat, bcs, symmetric_bcs = symmetric_bcs)
                self.__cache[key] = mat
            else:
                cache_info("Using cached assembled form with rank 2, with boundary conditions", dolfin.info_green)

        return self.__cache[key]

    def info(self):
        """
        Print some cache status information.
        """

        counts = [0, 0, 0]
        for key in self.__cache.keys():
            counts[form_rank(key[0])] += 1

        dolfin.info("Assembly cache status:")
        for i in range(3):
            dolfin.info("Pre-assembled rank %i forms: %i" % (i, counts[i]))

        return

    def clear(self, *args):
        """
        Clear the cache. If arguments are supplied, clear only the cached assembled
        Form s which depend upon the supplied Constant s or Function s.
        """

        if len(args) == 0:
            self.__cache = {}
        else:
            for dep in args:
                if not isinstance(dep, (dolfin.Constant, dolfin.Function)):
                    raise InvalidArgumentException("Arguments must be Constant s or Function s")

            for dep in args:
                for key in list(self.__cache.keys()):
                    form = key[0]
                    if dep in ufl.algorithms.extract_coefficients(form):
                        del(self.__cache[key])

        return

class SolverCache:
    """
    A cache of LUSolver s and KrylovSolver s. The linear_solver method can be used
    to return an LUSolver or KrylovSolver suitable for solving an equation with
    the supplied rank 2 Form defining the LHS matrix.
    """

    def __init__(self):
        self.__cache = OrderedDict()

        return

    def __del__(self):
        for key in list(self.__cache.keys()):
            del(self.__cache[key])

        return

    def linear_solver(self, form, linear_solver_parameters,
      pre_assembly_parameters = None,
      static = None,
      bcs = [], symmetric_bcs = False,
      a = None):
        """
        Return a linear solver suitable for solving an equation with the supplied
        rank 2 Form defining the LHS. If such a linear solver exists in the cache,
        return the cached linear solver.

        Arguments:
          form: The form defining the matrix.
          linear_solver_parameters: Linear solver parameters.
          bcs: Dirichlet BCs applied to the matrix.
          symmetric_bcs: Whether the Dirichlet BCs are applied so as to yield a
            symmetric matrix.
        and then either:
          pre_assembly_parameters: Pre-assembly parameters used to pre-assemble the
            form. Optional. Use an empty dictionary to denote the use of default
            pre-assembly parameters.
          static: Whether the form is static. Defaults to False if not supplied.
        or:
          a: A GenericMatrix resulting from the assembly of the form.

        This has three intended use cases:
          1. Pre-assembled form:
               linear_solver = linear_solver_cache.linear_solver(form,
                 linear_solver_parameters, pre_assembly_parameters,
                 static = static,
                 bcs = bcs, symmetric_bcs = symmetric_bcs)
             If the default pre-assembly parameters are used, then an empty
             dictionary should be passed as the third argument. Assemble the form
             using:
               pa_form = PAForm(form,
                 pre_assembly_parameters = pre_assembly_parameters)
               a = assemble(pa_form, ...)
               apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
          2. Cached matrix:
               a = assembly_cache.assemble(form, ...)
               apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
               linear_solver = linear_solver_cache.linear_solver(form,
                 linear_solver_parameters,
                 bcs = bcs, symmetric_bcs = symmetric_bcs,
                 a = a)
          3. Custom assembled form:
               linear_solver = linear_solver_cache.linear_solver(form,
                 linear_solver_parameters,
                 bcs = bcs, symmetric_bcs = symmetric_bcs)
             and then assemble the form using:
               a = assemble(form, ...)
               apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
        """

        def expanded_linear_solver_parameters(form, linear_solver_parameters, static, bcs, symmetric_bcs):
            if static:
                default = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True},
                           "krylov_solver":{}}
                if (len(bcs) == 0 or symmetric_bcs) and is_self_adjoint_form(form):
                    default["lu_solver"]["symmetric"] = True
                linear_solver_parameters = expand_linear_solver_parameters(linear_solver_parameters,
                  default_linear_solver_parameters = default)
            else:
                default = {"lu_solver":{"reuse_factorization":False, "same_nonzero_pattern":False},
                           "krylov_solver":{}}
                if (len(bcs) == 0 or symmetric_bcs) and is_self_adjoint_form(form):
                    default["lu_solver"]["symmetric"] = True
                linear_solver_parameters = expand_linear_solver_parameters(linear_solver_parameters,
                  default_linear_solver_parameters = default)

                static_parameters = False
                if linear_solver_parameters["linear_solver"] in ["direct", "lu"] or dolfin.has_lu_solver_method(linear_solver_parameters["linear_solver"]):
                    static_parameters = linear_solver_parameters["lu_solver"]["reuse_factorization"] or \
                                        linear_solver_parameters["lu_solver"]["same_nonzero_pattern"]
                else:
                    pass
                if static_parameters:
                    raise ParameterException("Non-static solve supplied with static linear solver parameters")

            return linear_solver_parameters

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a rank 2 Form")
        elif not form_rank(form) == 2:
            raise InvalidArgumentException("form must be a rank 2 Form")
        if not isinstance(linear_solver_parameters, dict):
            raise InvalidArgumentException("linear_solver_parameters must be a dictionary")

        if a is None:
            if static is None:
                static = False

            if not pre_assembly_parameters is None and not isinstance(pre_assembly_parameters, (dolfin.Parameters, dict)):
                raise InvalidArgumentException("pre_assembly_parameters must be None, a Parameters, or dictionary")
            if not isinstance(bcs, list):
                raise InvalidArgumentException("bcs must be a list of DirichletBC s")
            for bc in bcs:
                if not isinstance(bc, dolfin.cpp.DirichletBC):
                    raise InvalidArgumentException("bcs must be a list of DirichletBC s")

            linear_solver_parameters = expanded_linear_solver_parameters(form, linear_solver_parameters, static, bcs, symmetric_bcs)
            if not pre_assembly_parameters is None:
                npre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"].copy()
                npre_assembly_parameters.update(pre_assembly_parameters)
                pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)

            key = (form_key(form, static = static),
                   parameters_key(linear_solver_parameters),
                   None if pre_assembly_parameters is None else parameters_key(pre_assembly_parameters),
                   bc_key(bcs, symmetric_bcs),
                   None)
        else:
            if not isinstance(a, dolfin.GenericMatrix):
                raise InvalidArgumentException("a must be a GenericMatrix")

            if not pre_assembly_parameters is None:
                raise InvalidArgumentException("Cannot supply pre_assembly_parameters argument if a GenericMatrix is supplied")
            if not static is None:
                raise InvalidArgumentException("Cannot supply static argument if a GenericMatrix is supplied")

            static = True

            linear_solver_parameters = expanded_linear_solver_parameters(form, linear_solver_parameters, True, bcs, symmetric_bcs)

            key = (form_key(form),
                   parameters_key(linear_solver_parameters),
                   None,
                   bc_key(bcs, symmetric_bcs),
                   a.id())

        if not key in self.__cache:
            if static:
                cache_info("Creating new static linear solver", dolfin.info)
            else:
                cache_info("Creating new non-static linear solver", dolfin.info)
            self.__cache[key] = LinearSolver(linear_solver_parameters)
        else:
            if static:
                cache_info("Using cached static linear solver", dolfin.info)
            else:
                cache_info("Using cached non-static linear solver", dolfin.info)
        return self.__cache[key]

    def clear(self, *args):
        """
        Clear the cache. If arguments are supplied, clear only the linear solvers
        associated with Form s which depend upon the supplied Constant s or
        Function s.
        """

        if len(args) == 0:
            for key in list(self.__cache.keys()):
                del(self.__cache[key])
        else:
            for dep in args:
                if not isinstance(dep, (dolfin.Constant, dolfin.Function)):
                    raise InvalidArgumentException("Arguments must be Constant s or Function s")

            for key in list(self.__cache.keys()):
                form = key[0]
                if isinstance(form, ufl.form.Form) and dep in ufl.algorithms.extract_coefficients(form):
                    del(self.__cache[key])

        return

# Default assembly and linear solver caches.
assembly_cache = AssemblyCache()
linear_solver_cache = SolverCache()
def clear_caches(*args):
    """
    Clear the default assembly and linear solver caches. If arguments are
    supplied, clear only cached data associated with Form s which depend upon the
    supplied Constant s or Function s.
    """

    assembly_cache.clear(*args)
    linear_solver_cache.clear(*args)

    return
