#!/usr/bin/env python3

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard
# Copyright (C) 2008-2013 Martin Sandve Alnes
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

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard from FFC file
# ffc/analysis.py, bzr trunk 1839
# Code first added: 2012-12-06

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file
# ufl/algorithms/ad.py, bzr 1.1.x branch 1484
# Code first added: 2013-01-18

# Code derived from ffc/analysis.py in FFC 2016.1.0 added on 2016-10-07.
# Copyright information from ffc/analysis.py:
# 
# Copyright (C) 2007-201r Anders Logg and Kristian B. Oelgaard
#
# Modified by Marie E. Rognes, 2010
# Modified by Martin Alnaes, 2013-2014
#
# Code further checked against ffc/analysis.py in FFC 2017.1.0 on 2017-12-08.
# Copyright information from ffc/analysis.py:
#
# Copyright (C) 2007-2016 Anders Logg, Martin Alnaes, Kristian B. Oelgaard,
# and others

import copy

import dolfin
import ffc
import numpy
import ufl

from .exceptions import *
from .versions import *

__all__ = \
  [
    "apply_bcs",
    "assemble_symmetric_bcs",
    "differentiate_expr",
    "enforce_bcs",
    "extract_test_and_trial",
    "evaluate_expr",
    "expand",
    "expand_expr",
    "expand_linear_solver_parameters",
    "form_quadrature_degree",
    "form_rank",
    "is_empty_form",
    "is_r0_function",
    "is_r0_function_space",
    "is_self_adjoint_form",
    "lumped_mass"
  ]

def form_quadrature_degree(form):
    """
    Determine the quadrature degree with which the supplied Form is to be
    assembled. Return the default quadrature degree if one is set, or return the
    quadrature degree that would be selected by FFC. The final case duplicates the
    internal behaviour of FFC.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    if not dolfin.parameters["form_compiler"]["quadrature_degree"] is None \
      and dolfin.parameters["form_compiler"]["quadrature_degree"] >= 0:
        quadrature_degree = dolfin.parameters["form_compiler"]["quadrature_degree"]
    else:
        # This is based upon code from _analyze_form and
        # _attach_integral_metadata in analysis.py, FFC bzr trunk revision 1761
        #
        # Updated on 2016-10-07 using code from ffc/analysis.py in FFC 2016.1.0
        # Updated on 2017-12-08 and 2017-12-11 after checking against code from ffc/analysis.py in FFC 2017.1.0
        
        parameters = {}
        def copy_parameters(new, old):
          for key, value in old.items():
            if isinstance(value, (dict, dolfin.Parameters)):
              new[key] = {}
              copy_parameters(new[key], value)
            else:
              new[key] = value
        copy_parameters(parameters, dolfin.parameters["form_compiler"])
        
        form = copy.copy(form)
        form_data = ffc.analysis._analyze_form(form, parameters)
        quadrature_degree = -1
        for integral_data in form_data.integral_data:        
            for integral in integral_data.integrals:
                integral_metadata = copy.deepcopy(parameters)
                if integral.metadata():
                    integral_metadata.update(integral.metadata())
                quadrature_degree = max(quadrature_degree, ffc.analysis._autoselect_quadrature_degree(integral_metadata, integral, form_data))
        
        # End of code from derived from FFC

    return quadrature_degree

def form_rank(form):
    """
    Return the rank of the supplied Form.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    return len(ufl.algorithms.extract_arguments(form))

def is_r0_function(fn):
    """
    Return whether the supplied Function is R0 (i.e. a Real over the mesh).
    """

    if not isinstance(fn, dolfin.Function):
        raise InvalidArgumentException("fn must be a Function")

    return is_r0_function_space(fn.function_space())

def is_r0_function_space(space):
    """
    Return whether the supplied FunctionSpace is R0 (i.e. a Real over the mesh).
    """

    if not isinstance(space, dolfin.cpp.function.FunctionSpace):
        raise InvalidArgumentException("space must be a FunctionSpace")

    e = space.ufl_element()
    return e.family() == "Real" and e.degree() == 0

def evaluate_expr(expr, copy = False):
    """
    Evaluate the supplied expression, and return either a float or GenericVector.
    If copy is False then an existing GenericVector may be returned -- it is
    expected in this case that the return value will never be modified.
    """

    if not isinstance(expr, ufl.core.expr.Expr):
        raise InvalidArgumentException("expr must be an Expr")

    if isinstance(expr, ufl.algebra.Product):
        ops = expr.ufl_operands
        assert(len(ops) > 0)
        val = evaluate_expr(ops[0], copy = copy or len(ops) > 1)
        for op in ops[1:]:
            nval = evaluate_expr(op)
            if not isinstance(nval, float) or not nval == 1.0:
                val *= nval
    elif isinstance(expr, ufl.algebra.Sum):
        ops = expr.ufl_operands
        assert(len(ops) > 0)
        val = evaluate_expr(ops[0], copy = copy or len(ops) > 1)
        for op in ops[1:]:
            nval = evaluate_expr(op)
            if not isinstance(nval, float) or not nval == 0.0:
                val += nval
    elif isinstance(expr, ufl.algebra.Division):
        ops = expr.ufl_operands
        assert(len(ops) == 2)
        val = evaluate_expr(ops[0]) / evaluate_expr(ops[1])
    elif isinstance(expr, ufl.constantvalue.Zero):
        return 0.0
    elif isinstance(expr, dolfin.Function):
        if is_r0_function(expr):
            val = expr.vector().sum()
        else:
            val = expr.vector()
            if copy:
                val = val.copy()
    elif isinstance(expr, (dolfin.Constant, ufl.constantvalue.ConstantValue)):
        val = float(expr)
    elif isinstance(expr, ufl.differentiation.CoefficientDerivative):
        val = evaluate_expr(ufl.algorithms.expand_derivatives(expr))
    else:
        raise NotImplementedException("Expr type %s not implemented" % expr.__class__)

    return val

def differentiate_expr(expr, u, expand = True):
    """
    Wrapper for the UFL derivative function. This chooses an argument equal to
    Constant(1.0). Form s should be differentiated using the derivative function.
    """

    if not isinstance(expr, ufl.core.expr.Expr):
        raise InvalidArgumentException("expr must be an Expr")
    if isinstance(u, ufl.indexed.Indexed):
        op = u.ufl_operands
        assert(len(op) == 2)
        if not isinstance(op[0], (dolfin.Constant, dolfin.Function)):
            raise InvalidArgumentException("Invalid Indexed")
    elif not isinstance(u, (dolfin.Constant, dolfin.Function)):
        raise InvalidArgumentException("u must be an Indexed, Constant, or Function")

    if expr is u:
        der = ufl.constantvalue.IntValue(1)
    else:
        unity = dolfin.Constant(1.0)
        der = dolfin.replace(ufl.derivative(expr, u, argument = unity), {unity:ufl.constantvalue.IntValue(1)})

        if expand:
            # Based on code from expand_derivatives1 in UFL file ad.py, (see e.g. bzr
            # 1.1.x branch revision 1484)
            # Updated on 2017-12-08 (dim argument no longer needed)
            der = ufl.algorithms.expand_derivatives(der)

    return der

def expand_expr(expr):
    """
    Recursively expand the supplied Expr into the largest possible Sum.
    """

    if not isinstance(expr, ufl.core.expr.Expr):
        raise InvalidArgumentException("expr must be an Expr")

    if isinstance(expr, ufl.algebra.Sum):
        terms = []
        for term in expr.ufl_operands:
            terms += expand_expr(term)
        return terms
    elif isinstance(expr, ufl.algebra.Product):
        ops = expr.ufl_operands
        fact1 = ops[0]
        fact2 = ops[1]
        for op in ops[2:]:
            fact2 *= op
        fact1_terms = expand_expr(fact1)
        fact2_terms = expand_expr(fact2)
        terms = []
        for term1 in fact1_terms:
            for term2 in fact2_terms:
                terms.append(term1 * term2)
        return terms
    elif isinstance(expr, ufl.indexed.Indexed):
        ops = expr.ufl_operands
        assert(len(ops) == 2)
        return [ufl.indexed.Indexed(term, ops[1]) for term in expand_expr(ops[0])]
    elif isinstance(expr, ufl.tensors.ComponentTensor):
        ops = expr.ufl_operands
        assert(len(ops) == 2)
        return [ufl.tensors.ComponentTensor(term, ops[1]) for term in expand_expr(ops[0])]
    elif isinstance(expr, ufl.algebra.Division):
        ops = expr.ufl_operands
        assert(len(ops) == 2)
        return [ufl.algebra.Division(term, ops[1]) for term in expand_expr(ops[0])]
    elif isinstance(expr, ufl.restriction.PositiveRestricted):
        ops = expr.ufl_operands
        assert(len(ops) == 1)
        return [ufl.restriction.PositiveRestricted(term) for term in expand_expr(ops[0])]
    elif isinstance(expr, ufl.restriction.NegativeRestricted):
        ops = expr.ufl_operands
        assert(len(ops) == 1)
        return [ufl.restriction.NegativeRestricted(term) for term in expand_expr(ops[0])]
    elif isinstance(expr, ufl.differentiation.Grad):
        ops = expr.ufl_operands
        assert(len(ops) == 1)
        return [ufl.differentiation.Grad(term) for term in expand_expr(ops[0])]
    elif isinstance(expr, (ufl.tensoralgebra.Dot,
                           ufl.tensoralgebra.Inner,
                           ufl.differentiation.CoefficientDerivative,
                           ufl.differentiation.VariableDerivative)):
        return expand_expr(expand(expr))
    # Expr types white-list. These cannot be expanded.
    elif isinstance(expr, (ufl.constantvalue.ConstantValue,
                           ufl.argument.Argument,
                           dolfin.Expression,
                           dolfin.Function,
                           dolfin.Constant,
                           ufl.geometry.Circumradius,
                           ufl.algebra.Abs,
                           ufl.geometry.FacetNormal,
                           ufl.mathfunctions.Sqrt,
                           ufl.classes.Variable,
                           ufl.mathfunctions.Exp,
                           ufl.algebra.Power,
                           ufl.indexed.MultiIndex,
                           ufl.classes.Label)):
        return [expr]
    # Expr types grey-list. It might be possible to expand these, but just ignore
    # them at present.
    elif isinstance(expr, (ufl.tensors.ListTensor,
                           ufl.classes.Conditional,
                           ufl.indexsum.IndexSum)):
        return [expr]
    else:
        dolfin.warning("Expr type %s not expanded by expand_expr" % expr.__class__)
        return [expr]

def lumped_mass(space):
    """
    Return a linear form which can be assembled to yield a lumped mass matrix.
    """

    if not isinstance(space, dolfin.cpp.function.FunctionSpace):
        raise InvalidArgumentException("space must be a FunctionSpace")

    return dolfin.TestFunction(space) * dolfin.dx

def expand(form):
    """
    Expand the supplied Expr or Form. This attempts to yield a canonical form.
    """

    if not isinstance(form, (ufl.core.expr.Expr, ufl.form.Form)):
        raise InvalidArgumentException("form must be an Expr or Form")

    return ufl.algorithms.expand_indices(ufl.algorithms.expand_compounds(ufl.algorithms.expand_derivatives(form)))

def extract_test_and_trial(form):
    """
    Extract the test and trial function from a bi-linear form.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    args = ufl.algorithms.extract_arguments(form)
    if not len(args) == 2:
        raise InvalidArgumentException("form must be a bi-linear Form")
    test, trial = args
    if test.number() > trial.number():
        test, trial = trial, test
    assert(test.number() == trial.number() - 1)

    return test, trial

def is_self_adjoint_form(form):
    """
    Return True if the supplied Form is self-adjoint. May return false negatives.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    a_form = dolfin.adjoint(form)

    test, trial = extract_test_and_trial(form)
    a_test, a_trial = extract_test_and_trial(a_form)

    if not test.ufl_element() == a_trial.ufl_element():
        return False
    elif not trial.ufl_element() == a_test.ufl_element():
        return False
    else:
        return bool(expand(form) == expand(a_form))

def apply_bcs(a, bcs, L = None, symmetric_bcs = False):
    """
    Apply boundary conditions to the supplied LHS matrix and (optionally) RHS
    vector. If symmetric_bcs is true then the boundary conditions are applied so as
    to yield a symmetric matrix. If the boundary conditions are not homogeneous
    then a RHS vector should be supplied, although the lack of a RHS in this case
    is not treated as an error.
    """

    if not isinstance(a, dolfin.GenericMatrix):
        raise InvalidArgumentException("a must be a square GenericMatrix")
    elif not a.size(0) == a.size(1):
        raise InvalidArgumentException("a must be a square GenericMatrix")
    if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
        if not isinstance(bc, dolfin.cpp.DirichletBC):
            raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    if not L is None and not isinstance(L, dolfin.GenericVector):
        raise InvalidArgumentException("L must be a GenericVector")

    if len(bcs) == 0:
        return
    
    [bc.apply(*([a] if L is None else [a, L])) for bc in bcs]
    
    if symmetric_bcs:
        if L is None:
            L = a.factory().create_vector(a.mpi_comm())
            a.init_vector(L, 0)
            for bc in bcs:
                bc.zero_columns(a, L, 1.0)
        else:
            for bc in bcs:
                bc.zero_columns(a, L, 1.0)

    return

def enforce_bcs(x, bcs):
    """
    Enforce boundary conditions on the supplied GenericVector.
    """

    if not isinstance(x, dolfin.GenericVector):
        raise InvalidArgumentException("x must be a GenericVector")
    if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
        if not isinstance(bc, dolfin.cpp.DirichletBC):
            raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    for bc in bcs:
        bc.apply(x)

    return

def apply_default_parameters(parameters, default):
    """
    Return a parameters dictionary with a default values set.
    """

    lparameters = {}
    for key in parameters:
        if not isinstance(parameters[key], dict):
            lparameters[key] = parameters[key]
        else:
            lparameters[key] = apply_default_parameters(parameters[key], default.get(key, {}))
    for key in default:
        if not key in lparameters:
            lparameters[key] = copy.deepcopy(default[key])

    return lparameters

def expand_linear_solver_parameters(linear_solver_parameters, default_linear_solver_parameters = {}):
    """
    Return an expanded dictionary of linear solver parameters with all defaults
    explicitly specified. The optional default_linear_solver_parameters argument
    can be used to override global defaults.
    """

    if not isinstance(linear_solver_parameters, dict):
        raise InvalidArgumentException("linear_solver_parameters must be a dictionary")
    if not isinstance(default_linear_solver_parameters, dict):
        raise InvalidArgumentException("default_linear_solver_parameters must be a dictionary")

    linear_solver_parameters = apply_default_parameters(linear_solver_parameters, default_linear_solver_parameters)
    linear_solver_parameters = apply_default_parameters(linear_solver_parameters,
      {"linear_solver":"default",
       "preconditioner":"default",
       "lu_solver":dolfin.parameters["lu_solver"].to_dict(),
       "krylov_solver":dolfin.parameters["krylov_solver"].to_dict()
      })

    if linear_solver_parameters["linear_solver"] in ["default", "lu"] or dolfin.has_lu_solver_method(linear_solver_parameters["linear_solver"]):
        del(linear_solver_parameters["preconditioner"])
        del(linear_solver_parameters["krylov_solver"])
    else:
        del(linear_solver_parameters["lu_solver"])

    return linear_solver_parameters

def is_empty_form(form):
    """
    Return whether the supplied form is "empty" (i.e. contains no terms).
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    if len(form.integrals()) == 0:
        return True

    for integral in form.integrals():
        if not isinstance(integral.integrand(), ufl.constantvalue.Zero):
            return False
    return True
        
def assemble_symmetric_bcs(form, hbcs, tensor = None, form_compiler_parameters = {}):
    """
    Assemble the supplied rank 2 form, applying homogeneous boundary conditions
    so as to yield a symmetric matrix.
    """

    test, _ = extract_test_and_trial(form)
    shape = test.ufl_element().value_shape()
    dummy_rhs = dolfin.inner(test, dolfin.Constant(0.0 if len(shape) == 0 else numpy.zeros(shape, dtype = numpy.float64))) * dolfin.dx
    a, _ = dolfin.assemble_system(form, dummy_rhs, hbcs, A_tensor = tensor, form_compiler_parameters = form_compiler_parameters)
    return a
