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

import copy

import dolfin
import ufl

from exceptions import *
from fenics_utils import *

__all__ = \
  [
    "_KrylovSolver",
    "_LinearSolver",
    "_LUSolver",
    "_assemble",
    "_assemble_classes",
    "DirichletBC",
    "LinearSolver",
    "action",
    "adjoint",
    "assemble",
    "derivative",
    "homogenize",
    "lhs",
    "replace",
    "rhs"
  ]

# Assembly and linear solver functions and classes used in this module. These
# can be overridden externally.
_KrylovSolver = dolfin.KrylovSolver
_LinearSolver = dolfin.LinearSolver
_LUSolver = dolfin.LUSolver
_assemble = dolfin.assemble

class DirichletBC(dolfin.DirichletBC):
    """
    Wrapper for DOLFIN DirichletBC. Adds homogenized method.
    """

    def __init__(self, *args, **kwargs):
        dolfin.DirichletBC.__init__(self, *args, **kwargs)

        self.__hbc = None

        return

    def homogenized(self):
        """
        Return a homogenised version of this DirichletBC.
        """

        if self.__hbc is None:
            self.__hbc = dolfin.DirichletBC(self.function_space(), self.value(), *self.domain_args, method = self.method())
            self.__hbc.homogenize()
            if hasattr(self, "_time_static"):
                self.__hbc._time_static = self._time_static

        return self.__hbc

def homogenize(bc):
    """
    Return a homogenised version of the supplied DirichletBC.
    """

    if isinstance(bc, DirichletBC):
        return bc.homogenized()
    elif isinstance(bc, dolfin.cpp.DirichletBC):
        hbc = DirichletBC(bc.function_space(), bc.value(), *bc.domain_args, method = bc.method())
        hbc.homogenize()
        if hasattr(bc, "_time_static"):
            hbc._time_static = bc._time_static
    else:
        raise InvalidArgumentException("bc must be a DirichletBC")

    return hbc

def LinearSolver(*args, **kwargs):
    """
    Return a linear solver.

    Arguments: One of:
      1. Arguments as accepted by the DOLFIN LinearSolver constructor.
    or:
      2. A dictionary of linear solver parameters.
    """

    if not len(args) == 1 or not len(kwargs) == 0 or not isinstance(args[0], dict):
        return _LinearSolver(*args, **kwargs)
    linear_solver_parameters = args[0]

    linear_solver = "lu"
    pc = None
    kp = {}
    lp = {}
    for key in linear_solver_parameters:
        if key == "linear_solver":
            linear_solver = linear_solver_parameters[key]
        elif key == "preconditioner":
            pc = linear_solver_parameters[key]
        elif key == "krylov_solver":
            kp = linear_solver_parameters[key]
        elif key == "lu_solver":
            lp = linear_solver_parameters[key]
        elif key in ["print_matrix", "print_rhs", "reset_jacobian", "symmetric"]:
            raise NotImplementedException("Unsupported linear solver parameter: %s" % key)
        else:
            raise InvalidArgumentException("Unexpected linear solver parameter: %s" % key)

    if linear_solver in ["default", "direct", "lu"]:
        is_lu = True
        linear_solver = "default"
    elif linear_solver == "iterative":
        is_lu = False
        linear_solver = "gmres"
    else:
        is_lu = dolfin.has_lu_solver_method(linear_solver)

    if is_lu:
        linear_solver = _LUSolver(linear_solver)
        linear_solver.parameters.update(lp)
    else:
        if pc is None:
            linear_solver = _KrylovSolver(linear_solver)
        else:
            linear_solver = _KrylovSolver(linear_solver, pc)
        linear_solver.parameters.update(kp)

    return linear_solver

def adjoint(form, reordered_arguments = None, adjoint_arguments = None):
    """
    Wrapper for the DOLFIN adjoint function. Accepts the additional optional
    adjoint_arguments, which if supplied should be a tuple of Argument s
    corresponding to the adjoint test and trial functions. Correctly handles
    QForm s.
    """

    if adjoint_arguments is None:
        a_form = dolfin.adjoint(form, reordered_arguments = reordered_arguments)
    elif not reordered_arguments is None:
        raise InvalidArgumentException("Cannot supply both reordered_arguments and adjoint_arguments keyword arguments")
    else:
        if not len(adjoint_arguments) == 2 \
          or not isinstance(adjoint_arguments[0], ufl.argument.Argument) \
          or not isinstance(adjoint_arguments[1], ufl.argument.Argument):
            raise InvalidArgumentException("adjoint_arguments must be a pair of Argument s")

        a_test, a_trial = adjoint_arguments
        a_form = dolfin.adjoint(form)
        test, trial = extract_test_and_trial(a_form)

        if not test.element() == a_test.element() or not trial.element() == a_trial.element():
            raise InvalidArgumentException("Invalid adjoint_arguments")
        a_form = replace(a_form, {test:a_test, trial:a_trial})

    if isinstance(form, QForm):
        return QForm(a_form, quadrature_degree = form.quadrature_degree())
    else:
        return a_form

def replace(e, mapping):
    """
    Wrapper for the DOLFIN replace function. Correctly handles QForm s.
    """

    if not isinstance(mapping, dict):
        raise InvalidArgumentException("mapping must be a dictionary")

    if len(mapping) == 0:
        return e

    ne = dolfin.replace(e, mapping)
    if isinstance(e, QForm):
        return QForm(ne, quadrature_degree = form_quadrature_degree(e))
    else:
        return ne

def lhs(form):
    """
    Wrapper for the DOLFIN lhs function. Correctly handles QForm s.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    nform = dolfin.lhs(form)
    if isinstance(form, QForm):
        return QForm(nform, quadrature_degree = form_quadrature_degree(form))
    else:
        return nform

def rhs(form):
    """
    Wrapper for the DOLFIN rhs function. Correctly handles QForm s.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    nform = dolfin.rhs(form)
    if isinstance(form, QForm):
        return QForm(nform, quadrature_degree = form_quadrature_degree(form))
    else:
        return nform

def derivative(form, u, du = None, expand = True):
    """
    Wrapper for the DOLFIN derivative function. This attempts to select an
    appropriate du if one is not supplied. Correctly handles QForm s. By default
    the returned Form is first expanded using ufl.algorithms.expand_derivatives.
    This can be disabled if the optional expand argument is False.
    """

    if du is None:
        if isinstance(u, dolfin.Constant):
            du = dolfin.Constant(1.0)
        elif isinstance(u, dolfin.Function):
            rank = form_rank(form)
            if rank == 0:
                du = dolfin.TestFunction(u.function_space())
            elif rank == 1:
                du = dolfin.TrialFunction(u.function_space())

    der = dolfin.derivative(form, u, du = du)
    if expand:
        der = ufl.algorithms.expand_derivatives(der)

    if isinstance(form, QForm):
        return QForm(der, quadrature_degree = form.quadrature_degree())
    else:
        return der

def action(form, coefficient):
    """
    Wrapper for the DOLFIN action function. Correctly handles QForm s.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")
    if not isinstance(coefficient, dolfin.Function):
        raise InvalidArgumentException("coefficient must be a Function")

    nform = dolfin.action(form, coefficient)
    if isinstance(form, QForm):
        return QForm(nform, quadrature_degree = form_quadrature_degree(form))
    else:
        return nform

_assemble_classes = []
def assemble(*args, **kwargs):
    """
    Wrapper for the DOLFIN assemble function. Correctly handles PAForm s,
    TimeSystem s and QForm s.
    """

    if isinstance(args[0], QForm):
        if "form_compiler_parameters" in kwargs:
            raise InvalidArgumentException("Cannot supply form_compiler_parameters argument when assembling a QForm")
        return _assemble(form_compiler_parameters = args[0].form_compiler_parameters(), *args, **kwargs)
    elif isinstance(args[0], tuple(_assemble_classes)):
        return args[0].assemble(*args[1:], **kwargs)
    else:
        return _assemble(*args, **kwargs)
