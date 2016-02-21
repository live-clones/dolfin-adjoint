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

from collections import OrderedDict
import copy

import dolfin
import ufl

from caches import *
from equation_solvers import *
from exceptions import *
from fenics_overrides import *
from fenics_utils import *
from pre_assembled_forms import *
from statics import *
from time_levels import *
from time_functions import *

__all__ = \
  [
    "AdjointVariableMap",
    "PAAdjointSolvers",
    "TimeFunctional"
  ]

class AdjointVariableMap(object):
    """
    A map between forward and adjoint variables. Indexing into the
    AdjointVariableMap with a forward Function yields an associated adjoint
    Function, and similarly indexing into the AdjointVariableMap with an adjoint
    Function yields an associated forward Function. Allocates adjoint Function s
    as required.
    """

    def __init__(self):
        self.__a_tfns = {}
        self.__f_tfns = OrderedDict()
        self.__a_fns = {}
        self.__f_fns = OrderedDict()

        return

    def __getitem__(self, key):
        return self.__add(key)

    def __add(self, var):
        if isinstance(var, TimeFunction):
            if not var in self.__a_tfns:
                f_tfn = var
                a_tfn = AdjointTimeFunction(f_tfn)
                self.__a_tfns[f_tfn] = a_tfn
                self.__f_tfns[a_tfn] = f_tfn
                for level in f_tfn.all_levels():
                    self.__a_fns[f_tfn[level]] = a_tfn[level]
                    self.__f_fns[a_tfn[level]] = f_tfn[level]
            return self.__a_tfns[var]
        elif isinstance(var, AdjointTimeFunction):
            if not var in self.__f_tfns:
                f_tfn = var.forward()
                a_tfn = var
                self.__a_tfns[f_tfn] = a_tfn
                self.__f_tfns[a_tfn] = f_tfn
                for level in f_tfn.all_levels():
                    self.__a_fns[f_tfn[level]] = a_tfn[level]
                    self.__f_fns[a_tfn[level]] = f_tfn[level]
            return self.__f_tfns[var]
        elif isinstance(var, dolfin.Function):
            if is_static_coefficient(var):
                return var
            elif hasattr(var, "_time_level_data"):
                return self.__add(var._time_level_data[0])[var._time_level_data[1]]
            elif hasattr(var, "_adjoint_data"):
                if not var in self.__f_fns:
                    self.__a_fns[var._adjoint_data[0]] = var
                    self.__f_fns[var] = var._adjoint_data[0]
                return var._adjoint_data[0]
            else:
                if not var in self.__a_fns:
                    a_fn = dolfin.Function(name = "%s_adjoint" % var.name(), *[var.function_space()])
                    a_fn._adjoint_data = [var]
                    self.__a_fns[var] = a_fn
                    self.__f_fns[a_fn] = var
                return self.__a_fns[var]
        elif isinstance(var, dolfin.Constant):
            return var
        else:
            raise InvalidArgumentException("Argument must be an AdjointTimeFunction, TimeFunction, Function, or Constant")

    def zero_adjoint(self):
        """
        Zero all adjoint Function s,
        """

        for a_fn in self.__f_fns:
            if not hasattr(a_fn, "_time_level_data"):
                a_fn.vector().zero()
        for a_tfn in self.__f_tfns:
            a_tfn.zero()

        return

class TimeFunctional(object):
    """
    A template for a functional with an explicit time dependence.
    """

    def __init__(self):
        return

    def initialise(self, val = 0.0):
        """
        Initialise, with an initial functional value of val.
        """

        raise AbstractMethodException("initialise method not overridden")

    def addto(self, s):
        """
        Add to the functional at the end of timestep number s.
        """

        raise AbstractMethodException("addto method not overridden")

    def value(self):
        """
        Return the functional value.
        """

        raise AbstractMethodException("value method not overridden")

    def dependencies(self, s = None, non_symbolic = False):
        """
        Return the functional dependencies at the end of timestep number s. If
        non_symbolic is true, also return any other dependencies on which the value
        of the functional could depend at the end of timestep number s.
        """

        raise AbstractMethodException("dependencies method not overridden")

    def derivative(self, parameter, s):
        """
        Return the derivative of the functional with respect to the specified
        Constant of Function at the end of the timestep number s.
        """

        raise AbstractMethodException("derivative method not overridden")

class PAAdjointSolvers(object):
    """
    Defines a set of solves for adjoint equations, applying pre-assembly and
    linear solver caching optimisations. Expects as input a list of earlier
    forward equations and a list of later forward equations. If the earlier
    equations solve for {x_1, x_2, ...}, then the Function s on which the later
    equations depend should all be static or in the {x_1, x_2, ...}, although the
    failure of this requirement is not treated as an error.

    Constructor arguments:
      f_solves_a: Earlier time forward equations, as a list of AssignmentSolver s
        or EquationSolver s.
      f_solves_b: Later time forward equations, as a list of AssignmentSolver s
        or EquationSolver s.
      a_map: The AdjointVariableMap used to convert between forward and adjoint
        Function s.
    """

    def __init__(self, f_solves_a, f_solves_b, a_map):
        if not isinstance(f_solves_a, list):
            raise InvalidArgumentException("f_solves_a must be a list of AssignmentSolver s or EquationSolver s")
        for f_solve in f_solves_a:
            if not isinstance(f_solve, (AssignmentSolver, EquationSolver)):
                raise InvalidArgumentException("f_solves_a must be a list of AssignmentSolver s or EquationSolver s")
        if not isinstance(f_solves_b, list):
            raise InvalidArgumentException("f_solves_b must be a list of AssignmentSolver s or EquationSolver s")
        for f_solve in f_solves_b:
            if not isinstance(f_solve, (AssignmentSolver, EquationSolver)):
                raise InvalidArgumentException("f_solves_b must be a list of AssignmentSolver s or EquationSolver s")
        if not isinstance(a_map, AdjointVariableMap):
            raise InvalidArgumentException("a_map must be an AdjointVariableMap")

        # Reverse causality
        f_solves_a = copy.copy(f_solves_a);  f_solves_a.reverse()
        f_solves_b = copy.copy(f_solves_b);  f_solves_b.reverse()

        la_a_forms = []
        la_x = []
        la_L_forms = []
        la_L_as = []
        la_bcs = []
        la_solver_parameters = []
        la_pre_assembly_parameters = []
        la_keys = {}

        # Create an adjoint solve for each forward solve in f_solves_a, and add
        # the adjoint LHS
        for f_solve in f_solves_a:
            f_x = f_solve.x()
            a_x = a_map[f_x]
            a_space = a_x.function_space()
            assert(not a_x in la_keys)
            if isinstance(f_solve, AssignmentSolver):
                la_a_forms.append(None)
                la_bcs.append([])
                la_solver_parameters.append(None)
                la_pre_assembly_parameters.append(dolfin.parameters["timestepping"]["pre_assembly"].copy())
            else:
                assert(isinstance(f_solve, EquationSolver))
                f_a = f_solve.tangent_linear()[0]
                f_a_rank = form_rank(f_a)
                if f_a_rank == 2:
                    a_test, a_trial = dolfin.TestFunction(a_space), dolfin.TrialFunction(a_space)
                    a_a = adjoint(f_a, adjoint_arguments = (a_test, a_trial))
                    la_a_forms.append(a_a)
                    la_bcs.append(f_solve.hbcs())
                    la_solver_parameters.append(copy.deepcopy(f_solve.adjoint_solver_parameters()))
                else:
                    assert(f_a_rank == 1)
                    a_a = f_a
                    la_a_forms.append(a_a)
                    la_bcs.append(f_solve.hbcs())
                    la_solver_parameters.append(None)
                la_pre_assembly_parameters.append(f_solve.pre_assembly_parameters().copy())
            la_x.append(a_x)
            la_L_forms.append(None)
            la_L_as.append([])
            la_keys[a_x] = len(la_x) - 1

        # Add adjoint RHS terms corresponding to terms in each forward solve in
        # f_solves_a and f_solves_b
        for f_solve in f_solves_a + f_solves_b:
            f_x = f_solve.x()
            a_dep = a_map[f_x]
            if isinstance(f_solve, AssignmentSolver):
                f_rhs = f_solve.rhs()
                if isinstance(f_rhs, ufl.expr.Expr):
                    # Adjoin an expression assignment RHS
                    for f_dep in ufl.algorithms.extract_coefficients(f_rhs):
                        if isinstance(f_dep, dolfin.Function):
                            a_x = a_map[f_dep]
                            a_rhs = differentiate_expr(f_rhs, f_dep) * a_dep
                            if a_x in la_keys and not isinstance(a_rhs, ufl.constantvalue.Zero):
                                la_L_as[la_keys[a_x]].append(a_rhs)
                else:
                    # Adjoin a linear combination assignment RHS
                    for alpha, f_dep in f_rhs:
                        a_x = a_map[f_dep]
                        if a_x in la_keys:
                            la_L_as[la_keys[a_x]].append((alpha, a_dep))
            else:
                # Adjoin an equation RHS
                assert(isinstance(f_solve, EquationSolver))
                a_trial = dolfin.TrialFunction(a_dep.function_space())
                f_a_od = f_solve.tangent_linear()[1]
                for f_dep in f_a_od:
                    a_x = a_map[f_dep]
                    if a_x in la_keys:
                        a_test = dolfin.TestFunction(a_x.function_space())
                        a_key = la_keys[a_x]
                        a_form = -action(adjoint(f_a_od[f_dep], adjoint_arguments = (a_test, a_trial)), a_dep)
                        if la_L_forms[a_key] is None:
                            la_L_forms[a_key] = a_form
                        else:
                            la_L_forms[a_key] += a_form

        self.__a_map = a_map
        self.__a_a_forms = la_a_forms
        self.__a_x = la_x
        self.__a_L_forms = la_L_forms
        self.__a_L_as = la_L_as
        self.__a_bcs = la_bcs
        self.__a_solver_parameters = la_solver_parameters
        self.__a_pre_assembly_parameters = la_pre_assembly_parameters
        self.__a_keys = la_keys

        self.__functional = None
        self.reassemble()

        return

    def reassemble(self, *args):
        """
        Reassemble the adjoint solvers. If no arguments are supplied then all
        equations are re-assembled. Otherwise, only the LHSs or RHSs which depend
        upon the supplied Constant s or Function s are reassembled. Note that this
        does not clear the assembly or linear solver caches -- hence if a static
        Constant, Function, or DirichletBC is modified then one should clear the
        caches before calling reassemble on the PAAdjointSolvers.
        """

        def assemble_lhs(i):
            if self.__a_a_forms[i] is None:
                a_a = None
                a_solver = None
            else:
                a_a_rank = form_rank(self.__a_a_forms[i])
                if a_a_rank == 2:
                    static_bcs = n_non_static_bcs(self.__a_bcs[i]) == 0
                    static_form = is_static_form(self.__a_a_forms[i])
                    if len(self.__a_bcs[i]) > 0 and static_bcs and static_form:
                        a_a = assembly_cache.assemble(self.__a_a_forms[i],
                          bcs = self.__a_bcs[i], symmetric_bcs = self.__a_pre_assembly_parameters[i]["equations"]["symmetric_boundary_conditions"],
                          compress = self.__a_pre_assembly_parameters[i]["bilinear_forms"]["compress_matrices"])
                        a_solver = linear_solver_cache.linear_solver(self.__a_a_forms[i],
                          self.__a_solver_parameters[i],
                          bcs = self.__a_bcs[i], symmetric_bcs = self.__a_pre_assembly_parameters[i]["equations"]["symmetric_boundary_conditions"],
                          a = a_a)
                        a_solver.set_operator(a_a)
                    elif len(self.__a_bcs[i]) == 0 and static_form:
                        a_a = assembly_cache.assemble(self.__a_a_forms[i],
                          compress = self.__a_pre_assembly_parameters[i]["bilinear_forms"]["compress_matrices"])
                        a_solver = linear_solver_cache.linear_solver(self.__a_a_forms[i],
                          self.__a_solver_parameters[i],
                          a = a_a)
                        a_solver.set_operator(a_a)
                    else:
                        a_a = PABilinearForm(self.__a_a_forms[i], pre_assembly_parameters = self.__a_pre_assembly_parameters[i]["bilinear_forms"])
                        a_solver = linear_solver_cache.linear_solver(self.__a_a_forms[i],
                          self.__a_solver_parameters[i], self.__a_pre_assembly_parameters[i]["bilinear_forms"],
                          static = a_a.is_static() and static_bcs,
                          bcs = self.__a_bcs[i], symmetric_bcs = self.__a_pre_assembly_parameters[i]["equations"]["symmetric_boundary_conditions"])
                else:
                    assert(a_a_rank == 1)
                    assert(self.__a_solver_parameters[i] is None)
                    a_a = PALinearForm(self.__a_a_forms[i], pre_assembly_parameters = self.__a_pre_assembly_parameters[i]["linear_forms"])
                    a_solver = None
            return a_a, a_solver
        def assemble_rhs(i):
            if self.__a_L_forms[i] is None:
                return None
            else:
                return PALinearForm(self.__a_L_forms[i], pre_assembly_parameters = self.__a_pre_assembly_parameters[i]["linear_forms"])

        if len(args) == 0:
            la_a, la_solvers = [], []
            la_L = []
            for i in xrange(len(self.__a_x)):
                a_a, a_solver = assemble_lhs(i)
                a_L = assemble_rhs(i)
                la_a.append(a_a)
                la_solvers.append(a_solver)
                la_L.append(a_L)

            self.set_functional(self.__functional)
        else:
            la_a, la_solvers = copy.copy(self.__a_a), copy.copy(self.__a_solvers)
            la_L = copy.copy(self.__a_L)
            for i in xrange(len(self.__a_x)):
                for dep in args:
                    if not self.__a_a_forms[i] is None and dep in ufl.algorithms.extract_coefficients(self.__a_a_forms[i]):
                        la_a[i], la_solvers[i] = assemble_lhs(i)
                        break
                for dep in args:
                    if not self.__a_L_forms[i] is None and dep in ufl.algorithms.extract_coefficients(self.__a_L_forms[i]):
                        la_L[i] = assemble_rhs(i)
                        break

            if isinstance(self.__functional, ufl.form.Form):
                for dep in args:
                    if dep in ufl.algorithms.extract_coefficients(self.__functional):
                        self.set_functional(self.__functional)
                        break
            else:
                self.set_functional(self.__functional)

        self.__a_a, self.__a_solvers = la_a, la_solvers
        self.__a_L = la_L

        return

    def a_x(self):
        """
        Return the adjoint Function s being solved for.
        """

        return self.__a_x

    def solve(self):
        """
        Solve all adjoint equations.
        """

        for i in xrange(len(self.__a_x)):
            a_a = self.__a_a[i]
            a_x = self.__a_x[i]
            a_L = self.__a_L[i]
            a_L_as = self.__a_L_as[i]
            a_L_rhs = self.__a_L_rhs[i]
            a_bcs = self.__a_bcs[i]
            a_solver = self.__a_solvers[i]

            def evaluate_a_L_as(i):
                if isinstance(a_L_as[i], ufl.expr.Expr):
                    if is_r0_function(a_x):
                        L = evaluate_expr(a_L_as[i], copy = False)
                        if isinstance(L, dolfin.GenericVector):
                            l_L = L.sum()
                        else:
                            assert(isinstance(L, float))
                            l_L = L
                        L = a_x.vector().copy()
                        L[:] = l_L
                    else:
                        L = evaluate_expr(a_L_as[i], copy = True)
                        if isinstance(L, float):
                            l_L = L
                            L = a_x.vector().copy()
                            L[:] = l_L
                        else:
                            assert(isinstance(L, dolfin.GenericVector))
                else:
                    L = float(a_L_as[i][0]) * a_L_as[i][1].vector()
                return L
            def add_a_L_as(i, L):
                if isinstance(a_L_as[i], ufl.expr.Expr):
                    l_L = evaluate_expr(a_L_as[i], copy = False)
                    if is_r0_function(a_x):
                        if isinstance(l_L, dolfin.GenericVector):
                            l_L = l_L.sum()
                        else:
                            assert(isinstance(l_L, float))
                    if isinstance(l_L, dolfin.GenericVector):
                        L += l_L
                    else:
                        L.add_local(l_L * numpy.ones(L.local_range(0)[1] - L.local_range(0)[0]))
                        L.apply("insert")
                else:
                    L.axpy(float(a_L_as[i][0]), a_L_as[i][1].vector())
                return

            if a_L_rhs is None:
                if len(a_L_as) == 0:
                    if a_L is None:
                        if a_a is None or len(a_bcs) == 0:
                            a_x.vector().zero()
                            continue
                        else:
                            L = a_x.vector().copy()
                            L.zero()
                    else:
                        L = assemble(a_L, copy = len(a_bcs) > 0)
                else:
                    L = evaluate_a_L_as(0)
                    for i in xrange(1, len(a_L_as)):
                        add_a_L_as(i, L)
                    if not a_L is None:
                        L += assemble(a_L, copy = False)
            else:
                if isinstance(a_L_rhs, PAForm):
                    L = assemble(a_L_rhs, copy = len(a_bcs) > 0 or not a_L is None or len(a_L_as) > 0)
                else:
                    L = assemble(a_L_rhs)
                if not a_L is None:
                    L += assemble(a_L, copy = False)
                for i in xrange(len(a_L_as)):
                    add_a_L_as(i, L)

            if a_a is None:
                assert(len(a_bcs) == 0)
                assert(a_solver is None)
                a_x.vector()[:] = L
            elif a_solver is None:
                assert(a_a.rank() == 1)
                a_a = assemble(a_a, copy = False)
                assert(L.local_range() == a_a.local_range())
                a_x.vector().set_local(L.array() / a_a.array())
                a_x.vector().apply("insert")
                enforce_bcs(a_x.vector(), a_bcs)
            else:
                if isinstance(a_a, dolfin.GenericMatrix):
                    enforce_bcs(L, a_bcs)
                else:
                    a_a = assemble(a_a, copy = len(a_bcs) > 0)
                    apply_bcs(a_a, a_bcs, L = L, symmetric_bcs = self.__a_pre_assembly_parameters[i]["equations"]["symmetric_boundary_conditions"])
                    a_solver.set_operator(a_a)
                a_solver.solve(a_x.vector(), L)

        return

    def set_functional(self, functional):
        """
        Set a functional, defining associated adjoint RHS terms.
        """

        if functional is None:
            self.__a_L_rhs = [None for i in xrange(len(self.__a_x))]
            self.__functional = None
        elif isinstance(functional, ufl.form.Form):
            if not form_rank(functional) == 0:
                raise InvalidArgumentException("functional must be rank 0")

            a_rhs = OrderedDict()
            for f_dep in ufl.algorithms.extract_coefficients(functional):
                if is_static_coefficient(f_dep):
                    pass
                elif isinstance(f_dep, dolfin.Function):
                    a_x = self.__a_map[f_dep]
                    a_rhs[a_x] = derivative(functional, f_dep)
                elif isinstance(f_dep, (dolfin.Constant, dolfin.Expression)):
                    pass
                else:
                    raise DependencyException("Invalid dependency")

            self.__a_L_rhs = [None for i in xrange(len(self.__a_x))]
            for i, a_x in enumerate(a_rhs):
                if a_x in self.__a_keys:
                    self.__a_L_rhs[self.__a_keys[a_x]] = PALinearForm(a_rhs[a_x], pre_assembly_parameters = self.__a_pre_assembly_parameters[i]["linear_forms"])
            self.__functional = functional
        elif isinstance(functional, TimeFunctional):
            self.__a_L_rhs = [None for i in xrange(len(self.__a_x))]
            self.__functional = functional
        else:
            raise InvalidArgumentException("functional must be a Form or a TimeFunctional")

        return

    def update_functional(self, s):
        """
        Update the adjoint RHS associated with the functional at the end of timestep
        s.
        """

        if not isinstance(s, int) or s < 0:
            raise InvalidArgumentException("s must be a non-negative integer")

        if not isinstance(self.__functional, TimeFunctional):
            return

        a_rhs = OrderedDict()
        for f_dep in self.__functional.dependencies(s):
            if is_static_coefficient(f_dep):
                pass
            elif isinstance(f_dep, dolfin.Function):
                a_x = self.__a_map[f_dep]
                a_rhs[a_x] = self.__functional.derivative(f_dep, s)
            elif isinstance(f_dep, dolfin.Constant):
                pass
            else:
                raise DependencyException("Invalid dependency")

        self.__a_L_rhs = [None for i in xrange(len(self.__a_x))]
        for a_x in a_rhs:
            if not a_x in self.__a_keys:
                dolfin.warning("Missing functional dependency %s" % a_x.name())
            else:
                self.__a_L_rhs[self.__a_keys[a_x]] = a_rhs[a_x]

        return
