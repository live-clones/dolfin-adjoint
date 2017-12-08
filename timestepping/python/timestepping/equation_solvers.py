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

import copy

import dolfin
import ufl

from .exceptions import *
from .fenics_overrides import *
from .fenics_utils import *

__all__ = \
  [
    "AssignmentSolver",
    "EquationSolver",
    "LinearCombination"
  ]

class LinearCombination:
    """
    A linear combination.

    Constructor arguments: An arbitrary number of tuples, each of which contains
    two elements:
      (alpha, x)
    where alpha is one of:
        1. A float.
      or:
        2. An arbitrary Expr.
    and x is a Function. The supplied alphas are not checked to ensure that this
    really is a true linear combination.
    """

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, tuple) or not len(arg) == 2 \
              or not isinstance(arg[0], (float, ufl.core.expr.Expr)) or not isinstance(arg[1], dolfin.Function):
                raise InvalidArgumentException("Require tuples of (float or Expr, Function) as arguments")

        alpha = [arg[0] for arg in args]
        for i, lalpha in enumerate(alpha):
            if isinstance(lalpha, float):
                alpha[i] = ufl.constantvalue.FloatValue(lalpha)
        y = [arg[1] for arg in args]

        use_exp = False
        for lalpha in alpha:
            if not isinstance(lalpha, (ufl.constantvalue.FloatValue, dolfin.Constant)):
                use_exp = True
                break
        if use_exp:
            fl = ufl.constantvalue.Zero()
            for lalpha, ly in zip(alpha, y):
                fl += lalpha * ly
        else:
            fl = [(lalpha, ly) for lalpha, ly in zip(alpha, y)]

        self.__alpha = alpha
        self.__y = y
        self.__fl = fl

        return

    def dependencies(self, non_symbolic = False):
        """
        Return all dependencies associated with the LinearCombination. The
        optional non_symbolic argument has no effect.
        """

        deps = set(self.__y)
        for alpha in self.__alpha:
            deps.update(ufl.algorithms.extract_coefficients(alpha))

        return deps

    def nonlinear_dependencies(self):
        """
        Return all non-linear dependencies associated with the LinearCombination.
        """

        if isinstance(self.__fl, ufl.core.expr.Expr):
            nl_deps = set()
            for dep in self.dependencies():
                if isinstance(dep, dolfin.Function):
                    nl_deps.update(ufl.algorithms.extract_coefficients(differentiate_expr(self.__fl, dep)))
                elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
                    raise DependencyException("Invalid dependency")
            return nl_deps
        else:
            return set()

    def flatten(self):
        """
        Return a flattened version of the LinearCombination. If all alphas are
        floats or Constant s then return a list of (alpha, x) tuples. Otherwise
        return an Expr.
        """

        return self.__fl

    def solve(self, x):
        """
        Assign the given Function x to be equal to the value of the
        LinearCombination.
        """

        if not isinstance(x, dolfin.Function):
            raise InvalidArgumentException("x must be a Function")
        x = x.vector()

        if isinstance(self.__alpha[0], (ufl.constantvalue.FloatValue, dolfin.Constant)):
            lalpha = float(self.__alpha[0])
        else:
            lalpha = evaluate_expr(self.__alpha[0], copy = False)
        if isinstance(lalpha, float):
            x.zero()
            x.axpy(lalpha, self.__y[0].vector())
        else:
            assert(isinstance(lalpha, dolfin.GenericVector))
            x[:] = lalpha * self.__y[0].vector()[:]
        for alpha, y in zip(self.__alpha[1:], self.__y[1:]):
            if isinstance(alpha, (ufl.constantvalue.FloatValue, dolfin.Constant)):
                x.axpy(float(alpha), y.vector())
            else:
                lalpha = evaluate_expr(alpha, copy = False)
                if isinstance(lalpha, dolfin.GenericVector):
                    x[:] += lalpha * y.vector()
                else:
                    assert(isinstance(lalpha, float))
                    x.axpy(lalpha, y.vector())

        return

class AssignmentSolver:
    """
    A "solver" defining a direct assignment.

    Constructor arguments:
      y: The RHS, which must not depend on x. One of:
          1. A float
        or:
          2. A LinearCombination.
        or:
          3. An arbitrary Expr.
      x: A Function, which is assigned equal to the value of y in a "solve".
    """

    def __init__(self, y, x):
        if not isinstance(y, (int, float, LinearCombination, ufl.core.expr.Expr)) \
          and not is_general_constant(y):
            raise InvalidArgumentException("y must be an int, float, LinearCombination or Expr")
        if not isinstance(x, dolfin.Function):
            raise InvalidArgumentException("x must be a Function")

        if isinstance(y, (int, float, ufl.constantvalue.IntValue)):
            y = ufl.constantvalue.FloatValue(y)
        elif isinstance(y, dolfin.Function):
            if y is x:
                raise DependencyException("Assignment is non-linear")
        elif is_general_constant(y):
            pass
        elif isinstance(y, LinearCombination):
            if x in y.dependencies():
                raise DependencyException("Assignment is non-linear")
        else:
            assert(isinstance(y, ufl.core.expr.Expr))
            if x in ufl.algorithms.extract_coefficients(y):
                raise DependencyException("Assignment is non-linear")

        self.__y = y
        self.__x = x

        return

    def is_assembled(self):
        """
        Return whether the AssignmentSolver is assembled. Always true.
        """

        return True

    def assemble(self):
        """
        Assemble the AssignmentSolver. Alias to the reassemble method with no
        arguments.
        """

        self.reassemble()

        return

    def reassemble(self, *args):
        """
        Reassemble the AssignmentSolver. Has no effect.
        """

        return

    def dependencies(self, non_symbolic = False):
        """
        Return all dependencies of the AssignmentSolver, excluding x. The optional
        non_symbolic has no effect.
        """

        if isinstance(self.__y, ufl.constantvalue.FloatValue):
            return set()
        elif isinstance(self.__y, dolfin.Function) or is_general_constant(self.__y):
            return {self.__y}
        elif isinstance(self.__y, LinearCombination):
            return self.__y.dependencies(non_symbolic = non_symbolic)
        else:
            assert(isinstance(self.__y, ufl.core.expr.Expr))
            return set(ufl.algorithms.extract_coefficients(self.__y))

    def nonlinear_dependencies(self):
        """
        Return all non-linear dependencies of the AssignmentSolver.
        """

        if isinstance(self.__y, (ufl.constantvalue.FloatValue, dolfin.Function)) or is_general_constant(self.__y):
            return set()
        elif isinstance(self.__y, LinearCombination):
            return self.__y.nonlinear_dependencies()
        else:
            assert(isinstance(self.__y, ufl.core.expr.Expr))
            nl_deps = set()
            for dep in ufl.algorithms.extract_coefficients(self.__y):
                if isinstance(dep, dolfin.Function):
                    nl_deps.update(ufl.algorithms.extract_coefficients(differentiate_expr(self.__y, dep)))
                elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
                    raise DependencyException("Invalid dependency")
            return nl_deps

    def rhs(self):
        """
        Return the RHS, as either a tuple (FloatValue or Constant, Function) or an
        Expr.
        """

        if isinstance(self.__y, dolfin.Function):
            return [(ufl.constantvalue.FloatValue(1.0), self.__y)]
        elif isinstance(self.__y, LinearCombination):
            return self.__y.flatten()
        else:
            assert(isinstance(self.__y, ufl.core.expr.Expr))
            return self.__y

    def x(self):
        """
        Return the Function being solved for.
        """

        return self.__x

    def is_linear(self):
        """
        Return whether the AssignmentSolver is linear. Always true.
        """

        return True

    def solve(self):
        """
        Solve for x.
        """

        if isinstance(self.__y, ufl.constantvalue.FloatValue):
            self.__x.assign(dolfin.Constant(self.__y))
        elif isinstance(self.__y, (dolfin.Constant, dolfin.Function)):
            self.__x.assign(self.__y)
        elif is_general_constant(self.__y):
            self.__x.assign(dolfin.Constant([y_c for y_c in self.__y]))
        elif isinstance(self.__y, LinearCombination):
            self.__y.solve(self.__x)
        else:
            assert(isinstance(self.__y, ufl.core.expr.Expr))
            self.__x.vector()[:] = evaluate_expr(self.__y, copy = False)

        return

class EquationSolver:
    """
    A generic linear or non-linear equation solver.

    Constructor arguments:
      eq: The Equation being solved.
      x: The Function being solved for.
      bcs: A list of DirichletBC s.
      solver_parameters: A dictionary of solver parameters.
      adjoint_solver_parameters: A dictionary of linear solver parameters for an
        adjoint solve.
      pre_assembly_parameters: A dictionary of pre-assembly parameters.
    """

    def __init__(self, eq, x, bcs = [], solver_parameters = {}, adjoint_solver_parameters = None, pre_assembly_parameters = {}):
        if not isinstance(eq, ufl.equation.Equation):
            raise InvalidArgumentException("eq must be an Equation")
        if not isinstance(x, dolfin.Function):
            raise InvalidArgumentException("x must be a Function")
        if isinstance(bcs, dolfin.cpp.DirichletBC):
            bcs = [bcs]
        if not isinstance(bcs, list):
            raise InvalidArgumentException("bcs must be a DirichletBC or a list of DirichletBC s")
        for bc in bcs:
            if not isinstance(bc, dolfin.cpp.DirichletBC):
                raise InvalidArgumentException("bcs must be a DirichletBC or a list of DirichletBC s")
        if not isinstance(solver_parameters, dict):
            raise InvalidArgumentException("solver_parameters must be a dictionary")
        if not adjoint_solver_parameters is None and not isinstance(adjoint_solver_parameters, dict):
            raise InvalidArgumentException("adjoint_solver_parameters must be a dictionary")

        x_deps = set(ufl.algorithms.extract_coefficients(eq.lhs))
        if not is_zero_rhs(eq.rhs):
            x_deps.update(ufl.algorithms.extract_coefficients(eq.rhs))

        is_linear = not x in x_deps

        solver_parameters = copy.deepcopy(solver_parameters)
        if is_linear:
            linear_solver_parameters = solver_parameters
        else:
            for key in ["linear_solver", "preconditioner", "lu_solver", "krylov_solver"]:
                if key in solver_parameters:
                    raise ParameterException("Unexpected linear solver parameters")
            nl_solver = solver_parameters.get("nonlinear_solver", "newton")
            if nl_solver == "newton":
                if "snes_solver" in solver_parameters:
                    raise ParameterException("Unexpected SNES solver parameters")
                linear_solver_parameters = solver_parameters.get("newton_solver", {})
            elif nl_solver == "snes":
                if "newton_solver" in solver_parameters:
                    raise ParameterException("Unexpected Newton solver parameters")
                linear_solver_parameters = solver_parameters.get("snes_solver", {})
            else:
                raise ParameterException("Invalid non-linear solver: %s" % nl_solver)
            nlinear_solver_parameters = {}
            for key in linear_solver_parameters:
                if key in ["linear_solver", "preconditioner", "lu_solver", "krylov_solver"]:
                    nlinear_solver_parameters[key] = linear_solver_parameters[key]
            linear_solver_parameters = nlinear_solver_parameters;  del(nlinear_solver_parameters)
        linear_solver_parameters = copy.deepcopy(linear_solver_parameters)
        if adjoint_solver_parameters is None:
            adjoint_solver_parameters = linear_solver_parameters
        adjoint_solver_parameters = copy.deepcopy(adjoint_solver_parameters)
        npre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"].copy()
        npre_assembly_parameters.update(pre_assembly_parameters)
        pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)

        self.__eq = eq
        self.__x = x
        self.__bcs = copy.copy(bcs)
        self.__J = None
        self.__hbcs = None
        self.__solver_parameters = solver_parameters
        self.__linear_solver_parameters = linear_solver_parameters
        self.__adjoint_solver_parameters = adjoint_solver_parameters
        self.__pre_assembly_parameters = pre_assembly_parameters
        self.__x_deps = x_deps
        self.__is_linear = is_linear

        self.__tl = [None, False]

        return

    def is_assembled(self):
        """
        Return whether the EquationSolver is assembled. Always true.
        """

        return True

    def assemble(self):
        """
        Assemble the EquationSolver. Alias to the reassemble method with no
        arguments.
        """

        self.reassemble()

        return

    def reassemble(self, *args):
        """
        Reassemble the equation solver. Has no effect.
        """

        return

    def dependencies(self, non_symbolic = False):
        """
        Return Equation dependencies. The optional non_symbolic argument has no
        effect.
        """

        return self.__x_deps

    def nonlinear_dependencies(self):
        """
        Return non-linear dependencies of the equation, by returning the
        dependencies of the tangent linear equation.
        """

        d, od = self.tangent_linear()
        nl_deps = set(ufl.algorithms.extract_coefficients(d))
        for od_form in od.values():
            nl_deps.update(ufl.algorithms.extract_coefficients(od_form))

        return nl_deps

    def x(self):
        """
        Return the Function being solved for.
        """

        return self.__x

    def eq(self):
        """
        Return the Equation.
        """

        return self.__eq.lhs == self.__eq.rhs

    def bcs(self):
        """
        Return the DirichletBC s.
        """

        return self.__bcs

    def J(self):
        """
        Return the derivative of the residual with respect to x (the Jacobian).
        """

        if self.__J is None:
            if self.is_linear():
                form = dolfin.action(self.__eq.lhs, self.__x)
            else:
                form = self.__eq.lhs
            if not is_zero_rhs(self.__eq.rhs):
                form -= self.__eq.rhs
            self.__J = derivative(form, self.__x)

        return self.__J

    def hbcs(self):
        """
        Return a homogenised version of the DirichletBC s associated with the
        EquationSolver.
        """

        if self.__hbcs is None:
            self.__hbcs = [homogenize(bc) for bc in self.__bcs]

        return self.__hbcs

    def solver_parameters(self):
        """
        Return solver parameters.
        """

        return self.__solver_parameters

    def linear_solver_parameters(self):
        """
        Return linear solver parameters.
        """

        return self.__linear_solver_parameters

    def adjoint_solver_parameters(self):
        """
        Return adjoint solver parameters.
        """

        return self.__adjoint_solver_parameters

    def pre_assembly_parameters(self):
        """
        Return pre-assembly parameters.
        """

        return self.__pre_assembly_parameters

    def is_linear(self):
        """
        Return whether the Equation is written as a linear variational problem.
        """

        return self.__is_linear

    def __add_tangent_linear(self):
        a_d = None
        a_od = {}

        def add_lhs_dep(form, dep, x):
            add_rhs_dep(dolfin.action(-form, x), dep, x)
            return
        def add_rhs_dep(form, dep, x):
            if isinstance(dep, dolfin.Function):
                mat = derivative(-form, dep)
                if dep in a_od:
                    a_od[dep] += mat
                else:
                    a_od[dep] = mat
            elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
                raise DependencyException("Invalid dependency")
            return

        x = self.__x
        lhs, rhs = self.__eq.lhs, self.__eq.rhs

        if self.is_linear():
            a_d = lhs
            for dep in ufl.algorithms.extract_coefficients(lhs):
                if dep is x:
                    raise DependencyException("Invalid non-linear solve")
                add_lhs_dep(lhs, dep, x)
            if not is_zero_rhs(rhs):
                for dep in ufl.algorithms.extract_coefficients(rhs):
                    if dep is x:
                        raise DependencyException("Invalid non-linear solve")
                    add_rhs_dep(rhs, dep, x)
        else:
            for dep in ufl.algorithms.extract_coefficients(lhs):
                add_rhs_dep(-lhs, dep, x)
            if not is_zero_rhs(rhs):
                for dep in ufl.algorithms.extract_coefficients(rhs):
                    add_rhs_dep(rhs, dep, x)

            if not x in a_od:
                raise DependencyException("Missing LHS")
            a_d = a_od[x]
            del(a_od[x])

        self.__tl = [(a_d, a_od), True]

        return

    def tangent_linear(self):
        """
        Return the tangent linear equation. The tangent linear is returned in
        terms of blocks, where each block is a Form that can be used to assemble
        a matrix block associated with the given dependency. The return value is
        a tuple (diagonal Form, off diagonal Form s), where the off diagonal Form s
        are a dictionary with dependencies as keys and Form s as values.
        """

        if not self.__tl[1]:
            self.__add_tangent_linear()

        return self.__tl[0]

    def derivative(self, parameter):
        """
        Return the derivative of the residual with respect to the supplied Constant
        or Function.
        """

        if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
            raise InvalidArgumentException("parameter must be a Constant or Function")

        if self.is_linear():
            if form_rank(self.__eq.lhs) == 1:
                if parameter is self.__x:
                    form = self.__eq.lhs
                elif parameter in ufl.algorithms.extract_coefficients(self.__eq.lhs):
                    raise NotImplementedException("General derivative for linear variational problem with rank 1 LHS not implemented")
                else:
                    form = ufl.form.Form([])
            else:
                form = dolfin.action(self.__eq.lhs, self.__x)
        else:
            form = self.__eq.lhs
        if not is_zero_rhs(self.__eq.rhs):
            form -= self.__eq.rhs

        return derivative(form, parameter)

    def solve(self):
        """
        Solve the equation. This calls the DOLFIN solve function.
        """

        if self.is_linear() and form_rank(self.__eq.lhs) == 1:
            raise NotImplementedException("Solve for linear variational problem with rank 1 LHS not implemented")

        if self.__J is None:
            dolfin.solve(self.__eq, self.__x, self.__bcs, solver_parameters = self.__solver_parameters)
        else:
            dolfin.solve(self.__eq, self.__x, self.__bcs, J = self.__J, solver_parameters = self.__solver_parameters)
        return
