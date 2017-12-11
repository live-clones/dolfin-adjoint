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

from collections import OrderedDict
import copy
from fractions import Fraction

import dolfin
import numpy
import scipy
import scipy.optimize
import ufl

from .caches import *
from .checkpointing import *
from .equation_solvers import *
from .exceptions import *
from .fenics_overrides import *
from .fenics_utils import *
from .pre_assembled_adjoint import *
from .pre_assembled_equations import *
from .pre_assembled_forms import *
from .statics import *
from .time_levels import *

__all__ = \
  [
    "AdjointModel",
    "ForwardModel",
    "ManagedModel",
    "TimeSystem"
  ]

class TimeSystem:
    """
    Used to register timestep equations.
    """

    def __init__(self):
        self.__deps = OrderedDict()
        self.__init_solves = OrderedDict()
        self.__solves = OrderedDict()
        self.__final_solves = OrderedDict()

        self.__x_tfns = [[], True]
        self.__tfns = [[], True]
        self.__sorted_solves = [[[], [], []], True]

        self.__update = None

        return

    def __process_deps(self, deps, level):
        proc_deps = set()
        for dep in deps:
            if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
                dep_level = dep._time_level_data[1]
                if isinstance(level, (int, Fraction)):
                    if not isinstance(dep_level, (int, Fraction)):
                        raise DependencyException("Inconsistent dependency type")
                elif not isinstance(level, dep_level.__class__):
                    raise DependencyException("Inconsistent dependency type")
                if dep_level > level:
                    raise DependencyException("Future dependency")
                proc_deps.add(dep)
        proc_deps = list(proc_deps)
        class DependencyComparison:
            def __init__(self, x):
                self.__x = x
        
            def __eq__(self, other):
                return self.cmp(other) == 0

            def __gt__(self, other):
                return self.cmp(other) > 0

            def __lt__(self, other):
                return self.cmp(other) < 0

            def __ne__(self, other):
                return not self == other

            def __ge__(self, other):
                return not self < other

            def __le__(self, other):
                return not self > other
        
            def x(self):
                return self.__x
        
            def cmp(self, y):
                x = self.x()
                y = y.x()
                if hasattr(x, "_time_level_data"):
                    if hasattr(y, "_time_level_data"):
                        x_tfn, x_level = x._time_level_data
                        y_tfn, y_level = y._time_level_data
                        if x_tfn > y_tfn:
                            return 1
                        elif x_tfn < y_tfn:
                            return -1
                        elif isinstance(x_level, (int, Fraction)):
                            if isinstance(y_level, (int, Fraction)):
                                if x_level > y_level:
                                    return 1
                                elif x_level < y_level:
                                    return -1
                                else:
                                    return 0
                            else:
                                assert(isinstance(y_level, (TimeLevel, FinalTimeLevel)))
                                return -1
                        elif isinstance(x_level, TimeLevel):
                            if isinstance(y_level, (int, Fraction)):
                                return 1
                            elif isinstance(y_level, TimeLevel):
                                if x_level > y_level:
                                    return 1
                                elif x_level < y_level:
                                    return -1
                                else:
                                    return 0
                            else:
                                assert(isinstance(y_level, FinalTimeLevel))
                                return -1
                        else:
                            assert(isinstance(x_level, FinalTimeLevel))
                            if isinstance(y_level, (int, Fraction, TimeLevel)):
                                return 1
                            else:
                                assert(isinstance(y_level, FinalTimeLevel))
                                if x_level > y_level:
                                    return 1
                                elif x_level < y_level:
                                    return -1
                                else:
                                    return 0
                    else:
                        return 1
                elif hasattr(y, "_time_level_data"):
                    return -1
                else:
                    return x.id() - y.id()
        proc_deps.sort(key = DependencyComparison)

        return proc_deps

    def add_assignment(self, y, x):
        """
        Add an assignment. The arguments match those accepted by the
        AssignmentSolver constructor.
        """

        if self.has_solve(x):
            raise StateException("Solve for %s already registered" % x.name())

        solve = AssignmentSolver(y, x)
        x_deps = solve.dependencies(non_symbolic = True)

        if not hasattr(x, "_time_level_data"):
            raise InvalidArgumentException("Missing time level data")
        last_past_level = x._time_level_data[0].last_past_level()
        level = x._time_level_data[1]

        for dep in x_deps:
            if dep is x:
                raise DependencyException("Assignment is non-linear")

        if isinstance(level, (int, Fraction)):
            if level > last_past_level.offset():
                raise TimeLevelException("Must initialise past timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__init_solves
        elif isinstance(level, TimeLevel):
            if level <= last_past_level:
                raise TimeLevelException("Must solve for future timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__solves
        elif isinstance(level, FinalTimeLevel):
            if level.offset() <= last_past_level.offset():
                raise TimeLevelException("Must solve for future timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__final_solves
        else:
            raise InvalidArgumentException("Invalid time level data")

        add_solve[x] = solve
        self.__deps[x] = deps

        self.__x_tfns[0], self.__x_tfns[1] = None, False
        self.__tfns[0], self.__tfns[1] = None, False
        self.__sorted_solves[0], self.__sorted_solves[1] = None, False

        return

    def add_solve(self, *args, **kwargs):
        """
        Add an equation solve.

        Arguments: One of:
            1. Arguments as accepted by the add_assignment method.
          or:
            2. Arguments as accepts by the PAEquationSolver constructor.
          or:
            3. An AssignmentSolver.
          or:
            4. An EquationSolver.
        """

        if len(args) == 2 and len(kwargs) == 0 and \
          (isinstance(args[0], (int, float, LinearCombination, ufl.core.expr.Expr)) or is_general_constant(args[0])) \
          and isinstance(args[1], dolfin.Function):
            self.add_assignment(args[0], args[1])
            return

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (AssignmentSolver, EquationSolver)):
            solve = args[0]
            x = solve.x()
            if self.has_solve(x):
                raise StateException("Solve for %s already registered" % x.name())
            x_deps = solve.dependencies(non_symbolic = True)
            if isinstance(args[0], AssignmentSolver):
                for dep in x_deps:
                    if dep is x:
                        raise DependencyException("Invalid non-linear solve")
        else:
            s_kwargs = copy.copy(kwargs)
            if "initial_guess" in s_kwargs:
                if not s_kwargs["initial_guess"] is None and not isinstance(s_kwargs["initial_guess"], dolfin.Function):
                    raise InvalidArgumentException("initial_guess must be a Function")
                initial_guess = s_kwargs["initial_guess"]
                del(s_kwargs["initial_guess"])
            else:
                initial_guess = None
            if "adjoint_solver_parameters" in s_kwargs:
                if not isinstance(s_kwargs["adjoint_solver_parameters"], dict):
                    raise InvalidArgumentException("adjoint_solver_parameters must be a dictionary")
                del(s_kwargs["adjoint_solver_parameters"])
            if "pre_assembly_parameters" in s_kwargs:
                if not isinstance(s_kwargs["pre_assembly_parameters"], dict):
                    raise InvalidArgumentException("pre_assembly_parameters must be a dictionary")
                del(s_kwargs["pre_assembly_parameters"])
            if "form_compiler_parameters" in s_kwargs:
                raise NotImplementedException("form_compiler_parameters argument not supported")

            eargs = dolfin.fem.solving._extract_args(*args, **s_kwargs)
            x = eargs[1]
            if self.has_solve(x):
                raise StateException("Solve for %s already registered" % x.name())
            eq = eargs[0]
            solve = copy.copy(args), copy.copy(kwargs)
            x_deps = set(ufl.algorithms.extract_coefficients(eq.lhs))
            if not is_zero_rhs(eq.rhs):
                x_deps.update(ufl.algorithms.extract_coefficients(eq.rhs))
            eq_lhs = eq.lhs

            if form_rank(eq_lhs) == 2:
                for dep in x_deps:
                    if dep is x:
                        raise DependencyException("Invalid non-linear solve")
            if not initial_guess is None:
                x_deps.add(initial_guess)

        if not hasattr(x, "_time_level_data"):
            raise InvalidArgumentException("Missing time level data")
        last_past_level = x._time_level_data[0].last_past_level()
        level = x._time_level_data[1]

        if isinstance(level, (int, Fraction)):
            if level > last_past_level.offset():
                raise TimeLevelException("Must initialise past timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__init_solves
        elif isinstance(level, TimeLevel):
            if level <= last_past_level:
                raise TimeLevelException("Must solve for future timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__solves
        elif isinstance(level, FinalTimeLevel):
            if level.offset() <= last_past_level.offset():
                raise TimeLevelException("Must solve for future timestep value")
            deps = self.__process_deps(x_deps, level)
            add_solve = self.__final_solves
        else:
            raise InvalidArgumentException("Invalid time level data")

        add_solve[x] = solve
        self.__deps[x] = deps

        self.__x_tfns[0], self.__x_tfns[1] = None, False
        self.__tfns[0], self.__tfns[1] = None, False
        self.__sorted_solves[0], self.__sorted_solves[1] = None, False

        return

    def has_solve(self, x):
        """
        Return whether a solve for the given Function is registered.
        """

        if not isinstance(x, dolfin.Function):
            raise InvalidArgumentException("x must be a Function")
        elif not hasattr(x, "_time_level_data"):
            raise InvalidArgumentException("x is missing time level data")

        level = x._time_level_data[1]
        if isinstance(level, (int, Fraction)):
            return x in self.__init_solves
        elif isinstance(level, TimeLevel):
            return x in self.__solves
        else:
            assert(isinstance(level, FinalTimeLevel))
            return x in self.__final_solves

    def remove_solve(self, *args):
        """
        Remove any solve for the given Function.
        """

        for x in args:
            if not isinstance(x, dolfin.Function):
                raise InvalidArgumentException("Require Function s as arguments")
            elif not hasattr(x, "_time_level_data"):
                raise InvalidArgumentException("x is missing time level data")

        for x in args:
            level = x._time_level_data[1]
            if isinstance(level, (int, Fraction)):
                if x in self.__init_solves:
                    del(self.__init_solves[x])
                    del(self.__deps[x])
            elif isinstance(level, TimeLevel):
                if x in self.__solves:
                    del(self.__solves[x])
                    del(self.__deps[x])
            else:
                assert(isinstance(level, FinalTimeLevel))
                if x in self.__final_solves:
                    del(self.__final_solves[x])
                    del(self.__deps[x])

        if len(args) > 0:
            self.__x_tfns[0], self.__x_tfns[1] = None, False
            self.__tfns[0], self.__tfns[1] = None, False
            self.__sorted_solves[0], self.__sorted_solves[1] = None, False

        return

    def check_dependencies(self):
        """
        Verify dependencies, including a check for circular dependencies.
        """

        checked = set()
        def check(fn, deps, solves, against = []):
            if fn in solves and not fn in checked:
                assert(hasattr(fn, "_time_level_data"))
                level = fn._time_level_data[1]
                for dep in deps[fn]:
                    if dep is fn or not dep in solves:
                        continue
                    dep_level = dep._time_level_data[1]
                    if isinstance(level, (int, Fraction)):
                        if not isinstance(dep_level, (int, Fraction)):
                            raise DependencyException("Inconsistent dependency type")
                    elif not isinstance(level, dep_level.__class__):
                        raise DependencyException("Inconsistent dependency type")
                    if dep_level > level:
                        raise DependencyException("Future dependency for Function %s" % fn.name())
                    elif dep in against:
                        raise DependencyException("Circular dependency for Function %s" % fn.name())
                    check(dep, deps, solves, against = against + [fn])
            checked.add(fn)
            return

        for fn in self.__init_solves:
            check(fn, self.__deps, self.__init_solves)
        for fn in self.__solves:
            check(fn, self.__deps, self.__solves)
        for fn in self.__final_solves:
            check(fn, self.__deps, self.__final_solves)

        return

    def check_function_levels(self):
        """
        Check that solves for all appropriate Function levels are defined. Emit
        a warning on failure.
        """

        for tfn in self.tfns():
            last_past_level = tfn.last_past_level()
            for level in tfn.levels():
                if level > last_past_level:
                    if not tfn[level] in self.__solves:
                        dolfin.warning("Missing time level %s solve for TimeFunction %s" % (level, tfn.name()))
                else:
                    if not tfn.has_level(level.offset()) or not tfn[level.offset()] in self.__init_solves:
                        dolfin.warning("Missing time level %i solve for TimeFunction %s" % (level.offset(), tfn.name()))

        return

    def set_update(self, update):
        """
        Register an update function. This is a callback:
          def update(s, cs = None):
            ...
            return
        where the update function is called at the start of timetep s and the
        optional cs is a list of Constant s,  or Function s, or Expression s to be
        updated. If cs is not supplied then all relevant updates should be
        performed.
        """

        if not callable(update):
            raise InvalidArgumentException("update must be a callable")

        self.__update = update

        return

    def x_tfns(self):
        """
        Return a list of Function s which are to be solved for.
        """

        if self.__x_tfns[1]:
            tfns = self.__x_tfns[0]
        else:
            tfns = set()
            for solves in [self.__init_solves, self.__solves, self.__final_solves]:
                for fn in solves:
                    tfns.add(fn._time_level_data[0])
            tfns = sorted(list(tfns))

            self.__x_tfns[0], self.__x_tfns[1] = tfns, True

        return tfns

    def tfns(self):
        """
        Return a list of all registered TimeFunction s.
        """

        if self.__tfns[1]:
            tfns = self.__tfns[0]
        else:
            tfns = set()
            for x in self.__deps:
                tfns.add(x._time_level_data[0])
                for dep in self.__deps[x]:
                    tfns.add(dep._time_level_data[0])
            tfns = sorted(list(tfns))

            self.__tfns[0], self.__tfns[1] = tfns, True

        return tfns

    def sorted_solves(self):
        """
        Return all solves. Applies dependency resolution. Returns a tuple of
        lists, (initial solves, timestep solves, final solves), where each list
        contains AssignmentSolver s, EquationSolver s, or a tuple of arguments which
        can be used to construct PAEquationSolver s.
        """

        if self.__sorted_solves[1]:
            init_solves, solves, final_solves = self.__sorted_solves[0]
        else:
            self.check_function_levels()
            self.check_dependencies()

            def sort_solves(fn, deps, solves, ssolves, added_solves):
                if fn in added_solves:
                    return
                for dep in deps[fn]:
                    if not dep is fn and dep in solves:
                        sort_solves(dep, deps, solves, ssolves, added_solves)
                added_solves.append(fn)
                ssolves.append(solves[fn])
                return

            added_solves = []
            init_solves = []
            solves = []
            final_solves = []
            for x in self.__init_solves:
                sort_solves(x, self.__deps, self.__init_solves, init_solves, added_solves)
            for x in self.__solves:
                sort_solves(x, self.__deps, self.__solves, solves, added_solves)
            for x in self.__final_solves:
                sort_solves(x, self.__deps, self.__final_solves, final_solves, added_solves)

            self.__sorted_solves[0], self.__sorted_solves[1] = [init_solves, solves, final_solves], True

        return copy.copy(init_solves), copy.copy(solves), copy.copy(final_solves)

    def assemble(self, *args, **kwargs):
        """
        Return a ForwardModel if adjoint is False, and a ManagedModel with an
        adjoint model enabled otherwise.

        Valid keyword arguments:
          adjoint (default false): If true then a ForwardModel is returned. If false
            then an adjoint model is derived and a ManagedModel is returned.
        All other arguments are passed directly to the ForwardModel or ManagedModel
        constructors.
        """

        kwargs = copy.copy(kwargs)
        if "adjoint" in kwargs:
            adjoint = kwargs["adjoint"]
            del(kwargs["adjoint"])
        else:
            adjoint = False

        if adjoint:
            return ManagedModel(self, *args, **kwargs)
        else:
            return ForwardModel(self, *args, **kwargs)

_assemble_classes.append(TimeSystem)

class ForwardModel:
    """
    Used to solve timestep equations with timestep specific optimisations applied.

    Constructor arguments:
      tsystem: A TimeSystem defining the timestep equations.
      initialise: Whether the initialise method is to be called.
      reassemble: Whether the reassemble methods of solvers defined by the
        TimeSystem should be called.
    """

    def __init__(self, tsystem, initialise = True, reassemble = False):
        if not isinstance(tsystem, TimeSystem):
            raise InvalidArgumentException("tsystem must be a TimeSystem")

        tfns = tsystem.tfns()
        init_solves, solves, final_solves = tsystem.sorted_solves()

        for i, solve in enumerate(init_solves):
            if isinstance(solve, (AssignmentSolver, EquationSolver)):
                if not solve.is_assembled():
                    solve.assemble()
                elif reassemble:
                    solve.reassemble()
            else:
                init_solves[i] = PAEquationSolver(*solve[0], **solve[1])
        for i, solve in enumerate(solves):
            if isinstance(solve, (AssignmentSolver, EquationSolver)):
                if not solve.is_assembled():
                    solve.assemble()
                elif reassemble:
                    solve.reassemble()
            else:
                solves[i] = PAEquationSolver(*solve[0], **solve[1])
        for i, solve in enumerate(final_solves):
            if isinstance(solve, (AssignmentSolver, EquationSolver)):
                if not solve.is_assembled():
                    solve.assemble()
                elif reassemble:
                    solve.reassemble()
            else:
                final_solves[i] = PAEquationSolver(*solve[0], **solve[1])

        self.__tfns = tfns
        self.__init_solves = init_solves
        self.__solves = solves
        self.__final_solves = final_solves
        self.__update = tsystem._TimeSystem__update
        self.__s = 0

        if initialise:
            self.initialise()

        return

    def tfns(self):
        """
        Return a list of all registered TimeFunction s.
        """

        return self.__tfns

    def initialise(self):
        """
        Solve initial equations and perform the initial variable cycle. Also reset
        the timestep counter.
        """

        self.timestep_update(s = 0)

#    dolfin.info("Performing forward initial solves")
        for solve in self.__init_solves:
            solve.solve()

#    dolfin.info("Performing forward initial cycle")
        for tfn in self.__tfns:
            tfn.initial_cycle()

        return

    def timestep_update(self, s = None, cs = None):
        """
        Call the update callback. s can be used to set the value of the timestep
        counter. Otherwise the timestep counter is incremented. The optional cs is a
        list of Constant s, Function s, or Expression s to be passed to the callback
        for update.
        """

        if not s is None:
            if not isinstance(s, int) or s < 0:
                raise InvalidArgumentException("s must be a non-negative integer")
        if not cs is None:
            if not isinstance(cs, (list, set)):
                raise InvalidArgumentException("cs must be a list of Constant s, Function s, or Expression s")
            for c in cs:
                if not isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)):
                    raise InvalidArgumentException("cs must be a list of Constant s, Function s, or Expression s")

        if not s is None:
            self.__s = s
        else:
            self.__s += 1
        if not self.__update is None:
            self.__update(self.__s, cs = cs)

        return

    def timestep_solve(self):
        """
        Perform the timestep solves.
        """

#    dolfin.info("Performing forward timestep")
        for solve in self.__solves:
            solve.solve()

        return

    def timestep_cycle(self, extended = True):
        """
        Perform the timestep cycle. If extended is true, use the extended cycle
        with Function aliasing. Otherwise use Function copying.
        """

#    dolfin.info("Performing forward timestep cycle")
        for tfn in self.__tfns:
            tfn.cycle(extended = extended)

        return

    def timestep(self, ns = 1):
        """
        Perform ns timesteps.
        """

        for i in range(ns):
            self.timestep_update()
            self.timestep_solve()
            self.timestep_cycle()

        return

    def finalise(self):
        """
        Solve final equations and perform the final variable cycle.
        """

        self.timestep_update(s = self.__s + 1)

#    dolfin.info("Performing forward final cycle")
        for tfn in self.__tfns:
            tfn.final_cycle()

#    dolfin.info("Performing forward final solves")
        for solve in self.__final_solves:
            solve.solve()

        return

    def reassemble(self, *args, **kwargs):
        """
        Call the reassemble method of all solves, with the supplied arguments. This
        first clears the relevant cache data if clear_caches is true.

        Valid keyword arguments:
          clear_caches (default true): Whether to clear caches.
        """

        lclear_caches = True
        for key in kwargs:
            if key == "clear_caches":
                lclear_caches = kwargs["clear_caches"]
            else:
                raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
        if lclear_caches:
            clear_caches(*args)
        for solve in self.__init_solves + self.__solves + self.__final_solves:
            solve.reassemble(*args)

        return

class AdjointModel:
    """
    Used to solve adjoint timestep equations with timestep specific optimisations
    applied. This assumes that forward model data is updated externally.

    Constructor arguments:
      forward: A ForwardModel defining the forward model.
      functional: A rank 0 form or a TimeFunctional defining the functional.
    """

    def __init__(self, forward, functional = None):
        if not isinstance(forward, ForwardModel):
            raise InvalidArgumentException("forward must be a ForwardModel")

        # Step 1: Set up the adjoint variable map
        a_map = AdjointVariableMap()

        # Step 2: Set up the forward variable cycle "solvers"
        f_tfns = forward.tfns()
        f_as_init_solves = []
        f_as_solves = []
        # A block identity operator. In the first adjoint timestep cycle levels
        # n + i and N + i are logically distinct, but the Function s are aliased to
        # each other. Replacing f_as_solves with f_as_id_solves effectively zeros
        # the action of the off-diagonal blocks associated with levels n + i.
        f_as_id_solves = []
        f_as_final_solves = []
        for f_tfn in f_tfns:
            init_cycle_map = f_tfn.initial_cycle_map()
            cycle_map = f_tfn.cycle_map()
            final_cycle_map = f_tfn.final_cycle_map()
            for level in init_cycle_map:
                f_as_init_solves.append(AssignmentSolver(f_tfn[init_cycle_map[level]], f_tfn[level]))
            for level in cycle_map:
                f_as_solves.append(AssignmentSolver(f_tfn[cycle_map[level]], f_tfn[level]))
                f_as_id_solves.append(AssignmentSolver(0.0, f_tfn[level]))
            for level in final_cycle_map:
                f_as_final_solves.append(AssignmentSolver(f_tfn[final_cycle_map[level]], f_tfn[level]))

        # Step 3: Adjoin the final solves
#    dolfin.info("Initialising adjoint initial solves")
        a_initial_solves = PAAdjointSolvers(forward._ForwardModel__final_solves, [], a_map)
#    dolfin.info("Initialising adjoint initial cycle 1")
        a_initial_cycle1 = PAAdjointSolvers(f_as_final_solves, forward._ForwardModel__final_solves, a_map)
#    dolfin.info("Initialising adjoint initial cycle 2")
        a_initial_cycle2 = PAAdjointSolvers(f_as_id_solves, f_as_final_solves, a_map)
#    dolfin.info("Initialising adjoint initial cycle 3")
        a_initial_cycle3 = PAAdjointSolvers(f_as_init_solves, f_as_final_solves, a_map)

        # Step 4: Adjoin the timestep
#    dolfin.info("Initialising adjoint timestep cycle")
        a_cycle = PAAdjointSolvers(f_as_solves, forward._ForwardModel__solves, a_map)
#    dolfin.info("Initialising adjoint timestep")
        a_solves = PAAdjointSolvers(forward._ForwardModel__solves, f_as_solves, a_map)

        # Step 5: Adjoin the initial solves
#    dolfin.info("Initialising adjoint final cycle")
        # For the final adjoint cycle and solves we need to know the first forward
        # solves for all time levels. If no forward timestep "solve" is specified
        # for a given level, then we need to know the forward timestep "cycle"
        # instead.
        f_init_dep_solves = copy.copy(forward._ForwardModel__solves)
        f_x = set()
        for solve in forward._ForwardModel__solves:
            f_x.add(solve.x())
        for f_tfn in f_tfns:
            cycle_map = f_tfn.cycle_map()
            for level in cycle_map:
                if not f_tfn[level] in f_x:
                    f_init_dep_solves.append(AssignmentSolver(f_tfn[cycle_map[level]], f_tfn[level]))
        a_final_cycle = PAAdjointSolvers(f_as_init_solves, f_init_dep_solves, a_map)
#    dolfin.info("Initialising adjoint final solves")
        a_final_solves = PAAdjointSolvers(forward._ForwardModel__init_solves, f_as_init_solves, a_map)

        self.__forward = forward
        self.__a_map = a_map
        self.__a_initial_solves = a_initial_solves
        self.__a_initial_cycle1 = a_initial_cycle1
        self.__a_initial_cycle2 = a_initial_cycle2
        self.__a_initial_cycle3 = a_initial_cycle3
        self.__a_cycle = a_cycle
        self.__a_solves = a_solves
        self.__a_final_cycle = a_final_cycle
        self.__a_final_solves = a_final_solves

        self.__s = 0

        self.set_functional(functional)

        return

    def a_map(self):
        """
        Return the AdjointVariableMap used by the AdjointModel.
        """

        return self.__a_map

    def initialise(self):
        """
        Solve initial adjoint equations.
        """

        if self.__functional is None:
            dolfin.warning("Running adjoint model with no functional defined")

        self.__a_map.zero_adjoint()

#    dolfin.info("Performing adjoint initial solves")
        self.__a_initial_solves.solve()
#    dolfin.info("Performing adjoint initial cycle 1")
        self.__a_initial_cycle1.solve()

        self.__s = 0

        return

    def timestep_cycle(self):
        """
        Perform the adjoint timestep cycle.
        """

        if self.__s == 0:
#      dolfin.info("Performing adjoint initial cycle 2")
            self.__a_initial_cycle2.solve()
        else:
#      dolfin.info("Performing adjoint timestep cycle")
            self.__a_cycle.solve()

        return

    def timestep_solve(self):
        """
        Perform the adjoint timestep solve.
        """

#    dolfin.info("Performing adjoint timestep")
        self.__a_solves.solve()
        self.__s += 1

        return

    def timestep(self):
        """
        Perform an adjoint timestep.
        """

        self.timestep_cycle()
        self.timesep_solve()

        return

    def final_cycle(self):
        """
        Perform the final adjoint cycle.
        """

        if self.__s == 0:
#      dolfin.info("Performing adjoint initial cycle 3")
            self.__a_initial_cycle3.solve()
        else:
#      dolfin.info("Performing adjoint final cycle")
            self.__a_final_cycle.solve()

        return

    def finalise(self):
        """
        Solve final adjoint equations.
        """

#    dolfin.info("Performing adjoint final solves")
        self.__a_final_solves.solve()

        return

    def set_functional(self, functional):
        """
        Set a functional, defining associated adjoint RHS terms.
        """

        if functional is None:
            self.__a_initial_solves.set_functional(None)
            self.__a_initial_cycle1.set_functional(None)
            self.__a_initial_cycle2.set_functional(None)
            self.__a_cycle.set_functional(None)
            self.__a_final_solves.set_functional(None)
        elif isinstance(functional, ufl.form.Form):
            if not form_rank(functional) == 0:
                raise InvalidArgumentException("functional must be rank 0")

            for f_dep in ufl.algorithms.extract_coefficients(functional):
                if isinstance(f_dep, dolfin.Function) and not is_static_coefficient(f_dep):
                    if not hasattr(f_dep, "_time_level_data"):
                        raise DependencyException("Missing time level data")
                    f_ftn, level = f_dep._time_level_data
                    if not isinstance(level, FinalTimeLevel):
                        raise DependencyException("Functional must depend only upon final time levels or static coefficients")

            self.__a_initial_solves.set_functional(functional)
            self.__a_initial_cycle1.set_functional(functional)
            self.__a_initial_cycle2.set_functional(None)
            self.__a_cycle.set_functional(None)
            self.__a_final_solves.set_functional(None)
        elif isinstance(functional, TimeFunctional):
            self.__a_initial_solves.set_functional(None)
            self.__a_initial_cycle1.set_functional(None)
            self.__a_initial_cycle2.set_functional(functional)
            self.__a_cycle.set_functional(functional)
            self.__a_final_solves.set_functional(functional)
        else:
            raise InvalidArgumentException("functional must be a Form or a TimeFunctional")

        self.__functional = functional

        return

    def update_functional(self, s):
        """
        Update the adjoint RHS associated with the functional at the end of timestep
        s.
        """

        if not isinstance(s, int) or s < 0:
            raise InvalidArgumentException("s must be a non-negative integer")

        if isinstance(self.__functional, TimeFunctional):
            if s == 0:
                self.__a_final_solves.update_functional(s)
            elif self.__s == 0:
                self.__a_initial_cycle2.update_functional(s)
            else:
                self.__a_cycle.update_functional(s)

        return

    def reassemble(self, *args, **kwargs):
        """
        Reassemble the adjoint solvers. If no arguments are supplied then all
        equations are re-assembled. Otherwise, only the LHSs or RHSs which depend
        upon the supplied Constant s or Function s are reassembled. This
        first clears the relevant cache data if clear_caches is true.

        Valid keyword arguments:
          clear_caches (default true): Whether to clear caches.
        """

        lclear_caches = True
        for key in kwargs:
            if key == "clear_caches":
                lclear_caches = kwargs["clear_caches"]
            else:
                raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
        if lclear_caches:
            clear_caches(*args)
        self.__a_initial_solves.reassemble(*args)
        self.__a_initial_cycle1.reassemble(*args)
        self.__a_initial_cycle2.reassemble(*args)
        self.__a_initial_cycle3.reassemble(*args)
        self.__a_cycle.reassemble(*args)
        self.__a_solves.reassemble(*args)
        self.__a_final_cycle.reassemble(*args)
        self.__a_final_solves.reassemble(*args)

        return

class ManagedModel:
    """
    Used to solve forward and adjoint timestep equations. This performs necessary
    storage and recovery management.

    Constructor arguments:
      tsystem: A TimeSystem defining the timestep equations.
      functional: A rank 0 form or a TimeFunctional defining the functional.
      disk_period: Data is written to disk every disk_period timesteps. In order
        to compute an adjoint solution data is restored from disk and the forward
        solution recomputed between the storage points. If disk_period is equal to
        None then the entire forward solution is stored in memory.
      initialise: Whether the initialise method is to be called.
      reassemble: Whether the reassemble methods of solvers defined by the
        TimeSystem should be called.
    """

    def __init__(self, tsystem, functional = None, disk_period = None, initialise = True, reassemble = False):
        if not isinstance(tsystem, TimeSystem):
            raise InvalidArgumentException("tsystem must be a TimeSystem")
        if not functional is None and not isinstance(functional, (ufl.form.Form, TimeFunctional)):
            raise InvalidArgumentException("functional must be a Form or a TimeFunctional")
        if not disk_period is None:
            if not isinstance(disk_period, int) or disk_period <= 0:
                raise InvalidArgumentException("disk_period must be a positive integer")

        forward = assemble(tsystem, adjoint = False, initialise = False, reassemble = reassemble)
        adjoint = AdjointModel(forward)

        init_cp_cs1 = set()
        init_cp_cs2 = set()
        for solve in forward._ForwardModel__init_solves:
            init_cp_cs1.add(solve.x())
            for dep in solve.dependencies(non_symbolic = True):
                if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
                    init_cp_cs1.add(dep)
        cp_cs = set()
        cp_f_tfn = set()
        for solve in forward._ForwardModel__solves:
            f_x = solve.x()
            cp_cs.add(f_x)
            cp_f_tfn.add(f_x._time_level_data[0])
            for dep in solve.dependencies(non_symbolic = True):
                if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
                    cp_cs.add(dep)
                    cp_f_tfn.add(dep._time_level_data[0])
        nl_cp_cs = set()
        update_cs = set()
        for solve in forward._ForwardModel__solves:
            for c in solve.nonlinear_dependencies():
                if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                    nl_cp_cs.add(c)
                elif isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)) and not is_static_coefficient(c):
                    update_cs.add(c)
        final_cp_cs = set()

        for f_tfn in forward.tfns():
            for level in f_tfn.initial_levels():
                init_cp_cs1.add(f_tfn[level])
                init_cp_cs2.add(f_tfn[level])
            if f_tfn in cp_f_tfn:
                for level in f_tfn.cycle_map().values():
                    cp_cs.add(f_tfn[level])
            for level in f_tfn.final_levels():
                final_cp_cs.add(f_tfn[level])

        self.__forward = forward
        self.__adjoint = adjoint
        self.__a_map = adjoint.a_map()
        self.__memory_checkpointer = MemoryCheckpointer()
        if disk_period is None:
            self.__disk_checkpointer = None
        else:
            self.__disk_checkpointer = DiskCheckpointer()
        self.__disk_period = disk_period
        self.__disk_m = -1
        self.__init_cp_cs1 = init_cp_cs1
        self.__init_cp_cs2 = init_cp_cs2
        self.__cp_cs = cp_cs
        self.__nl_cp_cs = nl_cp_cs
        self.__update_cs = update_cs
        self.__final_cp_cs = final_cp_cs
        self.__s = None
        self.__S = None

        self.set_functional(functional)

        if initialise:
            self.initialise()

        return

    def __timestep_checkpoint(self, s, cp_cs):
        if self.__disk_period is None:
            self.__memory_checkpointer.checkpoint(s, cp_cs)
        elif s < 0:
            self.__disk_checkpointer.checkpoint(s, cp_cs)
        else:
            m = s // self.__disk_period
            i = s % self.__disk_period
            if not m == self.__disk_m:
                self.__disk_checkpointer.checkpoint(m, cp_cs)
                self.__disk_m = m

        return

    def __restore_timestep_checkpoint(self, s, cp_cs = None, update_cs = None):
        if update_cs is None:
            def lupdate_cs(s):
                return None
        elif isinstance(update_cs, (list, set)):
            def lupdate_cs(s):
                return update_cs
        else:
            assert(callable(update_cs))
            lupdate_cs = update_cs

        if cp_cs is None:
            def lcp_cs(s):
                return self.__cp_cs
        elif isinstance(cp_cs, (list, set)):
            def lcp_cs(s):
                return cp_cs
        else:
            assert(callable(cp_cs))
            lcp_cs = cp_cs

        if self.__disk_period is None:
            if s >= 0:
                self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
                self.__memory_checkpointer.restore(s, cs = lcp_cs(s))
            else:
                self.__memory_checkpointer.restore(s)
        elif s < 0:
            self.__disk_checkpointer.restore(s)
        else:
            m = s // self.__disk_period
            i = s % self.__disk_period

            if (m, i) in self.__memory_checkpointer:
                self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
                self.__memory_checkpointer.restore((m, i))
            else:
                self.__clear_rerun_checkpoints()
                self.__disk_checkpointer.restore(m)
                self.__disk_m = m

                self.__memory_checkpointer.checkpoint((m, 0), lcp_cs(m * self.__disk_period))
                if m == self.__S // self.__disk_period:
                    for j in range(1, (self.__S % self.__disk_period) + 1):
                        ls = m * self.__disk_period + j
                        if m > 0 or j > 1:
                            self.__forward.timestep_cycle()
                        self.__forward.timestep_update(s = ls)
                        self.__forward.timestep_solve()
                        self.__memory_checkpointer.checkpoint((m, j), lcp_cs(ls))

                    if not i == self.__S % self.__disk_period:
                        self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
                        self.__memory_checkpointer.restore((m, i))
                else:
                    for j in range(1, self.__disk_period):
                        ls = m * self.__disk_period + j
                        if m > 0 or j > 1:
                            self.__forward.timestep_cycle()
                        self.__forward.timestep_update(s = ls)
                        self.__forward.timestep_solve()
                        self.__memory_checkpointer.checkpoint((m, j), lcp_cs(ls))

                    if not i == self.__disk_period - 1:
                        self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
                        self.__memory_checkpointer.restore((m, i))

        return

    def __clear_rerun_checkpoints(self):
        if not self.__disk_period is None:
            self.__memory_checkpointer.clear()
            self.__disk_m = -1

        return

    def __clear_timestep_checkpoints(self, keep = []):
        if self.__disk_period is None:
            self.__memory_checkpointer.clear(keep = keep)
        else:
            self.__disk_checkpointer.clear(keep = keep)
            self.__memory_checkpointer.clear()

        return

    def __verify_timestep_checkpoint(self, s, tolerance = 0.0):
        if self.__disk_period is None:
            self.__memory_checkpointer.verify(s, tolerance = tolerance)
        elif s < 0:
            self.__disk_checkpointer.verify(s, tolerance = tolerance)
        else:
            m = s // self.__disk_period
            i = s % self.__disk_period
            if i == 0:
                self.__disk_checkpointer.verify(m, tolerance = tolerance)
            if (m, i) in self.__memory_checkpointer:
                self.__memory_checkpointer.verify((m, i), tolerance = tolerance)

        return

    def a_map(self):
        """
        Return the AdjointVariableMap associated with the ManagedModel.
        """

        return self.__a_map

    def initialise(self):
        """
        Solve initial equations.
        """

        if not self.__S is None:
            raise StateException("Initialisation after finalisation")
        elif not self.__s is None:
            raise StateException("Multiple initialisations")

        self.__timestep_checkpoint(-1, self.__init_cp_cs1)
        self.__s = 0
        self.__forward.initialise()
        self.__timestep_checkpoint(-2, self.__init_cp_cs2)

#    self.__timestep_checkpoint(0, self.__cp_cs)

        if isinstance(self.__functional, TimeFunctional):
            self.__functional.initialise()
            self.__functional.addto(0)

        return

    def timestep(self, ns = 1):
        """
        Perform ns timesteps.
        """

        if self.__s is None:
            raise StateException("Timestep before initialisation")
        elif not self.__S is None:
            raise StateException("Timestep after finalisation")

        for i in range(ns):
            self.__s += 1
            self.__forward.timestep_update()
            self.__forward.timestep_solve()
            self.__timestep_checkpoint(self.__s, self.__cp_cs)
            self.__forward.timestep_cycle()
            if isinstance(self.__functional, TimeFunctional):
                self.__functional.addto(self.__s)

        return

    def finalise(self):
        """
        Solve final equations.
        """

        if self.__s is None:
            raise StateException("Finalisation before initialisation")
        elif not self.__S is None:
            raise StateException("Multiple finalisations")

        self.__timestep_checkpoint(-3, self.__cp_cs)
        self.__forward.finalise()
        self.__S = self.__s
        self.__timestep_checkpoint(-4, self.__final_cp_cs)

        return

    def verify_checkpoints(self, tolerance = 0.0):
        """
        Verify the reproducibility of stored forward model data to within the
        specified tolerance.
        """

        if self.__S is None:
            raise StateException("Attempting to rerun model before finalisation")

        self.__restore_timestep_checkpoint(-1)
        self.__forward.initialise()
        self.__verify_timestep_checkpoint(-2, tolerance = tolerance)

#    self.__verify_timestep_checkpoint(0, tolerance = tolerance)
        for i in range(self.__S):
            self.__forward.timestep_update()
            self.__forward.timestep_solve()
            self.__verify_timestep_checkpoint(i + 1, tolerance = tolerance)
            self.__forward.timestep_cycle()

        self.__verify_timestep_checkpoint(-3, tolerance = tolerance)
        self.__forward.finalise()
        self.__verify_timestep_checkpoint(-4, tolerance = tolerance)

        return

    def set_functional(self, functional):
        """
        Set a functional, defining associated adjoint RHS terms.
        """

        self.__adjoint.set_functional(functional)
        self.__functional = functional

        return

    def compute_functional(self, rerun_forward = False, recheckpoint = False):
        """
        Return the functional value, optionally re-running the forward model and
        optionally regenerating the forward model storage data.
        """

        if self.__functional is None:
            raise StateException("No functional defined")

        if rerun_forward:
            self.rerun_forward(recheckpoint = recheckpoint)

        if isinstance(self.__functional, ufl.form.Form):
            return assemble(self.__functional)
        else:
            return self.__functional.value()

    def rerun_forward(self, recheckpoint = False):
        """
        Re-run the forward model, optionally regenerating the forward model storage
        data.
        """

        if self.__S is None:
            raise StateException("Attempting to rerun model before finalisation")

        self.__restore_timestep_checkpoint(-1)
        if recheckpoint:
            self.__clear_timestep_checkpoints(keep = [-1])

        self.__forward.initialise()
        if recheckpoint:
            self.__timestep_checkpoint(-2, self.__init_cp_cs2)
#      self.__timestep_checkpoint(0, self.__cp_cs)
        if isinstance(self.__functional, TimeFunctional):
            self.__functional.initialise()
            self.__functional.addto(0)

        for i in range(self.__S):
            self.__forward.timestep_update()
            self.__forward.timestep_solve()
            if recheckpoint:
                self.__timestep_checkpoint(i + 1, self.__cp_cs)
            self.__forward.timestep_cycle()
            if isinstance(self.__functional, TimeFunctional):
                self.__functional.addto(i + 1)

        if recheckpoint:
            self.__timestep_checkpoint(-3, self.__cp_cs)
        self.__forward.finalise()
        if recheckpoint:
            self.__timestep_checkpoint(-4, self.__final_cp_cs)

        return

    def reassemble_forward(self, *args, **kwargs):
        """
        Reassemble the forward model. This first clears the relevant cache data if
        clear_caches is true.

        Valid keyword arguments:
          clear_caches (default true): Whether to clear caches.
        """

        self.__forward.reassemble(*args, **kwargs)

        return

    def reassemble_adjoint(self, *args, **kwargs):
        """
        Reassemble the adjoint model. This first clears the relevant cache data if
        clear_caches is true.

        Valid keyword arguments:
          clear_caches (default true): Whether to clear caches.
        """

        self.__adjoint.reassemble(*args, **kwargs)

        return

    def reassemble(self, *args, **kwargs):
        """
        Reassemble the forward and adjoint models. This first clears the relevant
        cache data if clear_caches is true.

        Valid keyword arguments:
          clear_caches (default true): Whether to clear caches.
        """

        lclear_caches = True
        for key in kwargs:
            if key == "clear_caches":
                lclear_caches = kwargs["clear_caches"]
            else:
                raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
        if lclear_caches:
            clear_caches(*args)
        self.reassemble_forward(clear_caches = False, *args)
        self.reassemble_adjoint(clear_caches = False, *args)

        return

    def compute_gradient(self, parameters, callback = None, project = False, project_solver_parameters = {}):
        """
        Compute the model constrained derivative of the registered functional with
        respect to the supplied Constant or Function, or list of Constant s or
        Function s. The optional callback function has the form:
          def callback(s):
            ...
            return
        and is called at the end of forward timestep s.

        The return value has the following type:
          1. parameters is a Constant: The derivative is returned as a Constant.
          2. parameters is a Function, and
             a. project is False: The derivative is returned as a GenericVector.
             b. project is True: The derivative is returned as a (GenericVector,
                Function) pair, where the Function contains the derivative projected
                onto the trial space of parameters. project_solver_parameters
                defines linear solver options for the associated mass matrix
                inversion.
          3. parameters is a list of Constant s or Function s: The derivative is
             returned as a list. If parameters[i] is a Constant, then element i
             of the returned list has a type as given by 1. above. If parameters[i]
             is a Function then element i of the returned list has a type given by
             2. above.
        """

        if isinstance(parameters, list):
            for parameter in parameters:
                if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
                    raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
        elif isinstance(parameters, (dolfin.Constant, dolfin.Function)):
            return self.compute_gradient([parameters], callback = callback, project = project, project_solver_parameters = project_solver_parameters)[0]
        else:
            raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
        if self.__S is None:
            raise StateException("Attempting to compute gradient before finalisation")
        if not callback is None and not callable(callback):
            raise InvalidArgumentException("callback must be a callable")
        if not project_solver_parameters is None and not isinstance(project_solver_parameters, dict):
            raise InvalidArgumentException("project_solver_parameters must be a dict")

        for parameter in parameters:
            if not is_static_coefficient(parameter):
                raise InvalidArgumentException("Differentiating with respect to non-static parameter")

        f_init_solves = self.__forward._ForwardModel__init_solves
        f_solves = self.__forward._ForwardModel__solves
        f_final_solves = self.__forward._ForwardModel__final_solves

        dFdm = {j:[OrderedDict() for i in range(len(parameters))] for j in range(-1, 2)}
        for i, parameter in enumerate(parameters):
            for j, solves in zip(range(-1, 2), [f_init_solves, f_solves, f_final_solves]):
                for f_solve in solves:
                    f_x = f_solve.x()
                    if isinstance(f_solve, AssignmentSolver):
                        f_rhs = f_solve.rhs()
                        if isinstance(f_rhs, ufl.core.expr.Expr):
                            f_der = differentiate_expr(f_rhs, parameter)
                            if not isinstance(f_der, ufl.constantvalue.Zero):
                                dFdm[j][i][f_x] = f_der
                        else:
                            f_der = []
                            for alpha, f_dep in f_rhs:
                                if parameter is alpha:
                                    f_der.append(f_dep)
                                elif parameter is f_dep:
                                    f_der.append(alpha)
                            if len(f_der) > 0:
                                dFdm[j][i][f_x] = f_der
                    else:
                        assert(isinstance(f_solve, EquationSolver))
                        f_der = f_solve.derivative(parameter)
                        if not is_empty_form(f_der):
                            rank = form_rank(f_der)
                            if rank == 1:
                                f_der = PAForm(f_der)
                            else:
                                assert(rank == 2)
                                f_der = PAForm(adjoint(f_der, \
                                  adjoint_arguments = (dolfin.TestFunction(parameter.function_space()), dolfin.TrialFunction(f_x.function_space()))))
                            dFdm[j][i][f_x] = f_der

        cp_cs = copy.copy(self.__nl_cp_cs)
        update_cs = copy.copy(self.__update_cs)
        for i in range(len(parameters)):
            for f_der in dFdm[0][i].values():
                if isinstance(f_der, PAForm):
                    for c in f_der.dependencies(non_symbolic = True):
                        if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                            cp_cs.add(c)
                        elif isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)) and not is_static_coefficient(c):
                            update_cs.add(c)
                elif isinstance(f_der, ufl.core.expr.Expr):
                    for c in ufl.algorithms.extract_coefficients(f_der):
                        if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                            cp_cs.add(c)
                        elif isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)) and not is_static_coefficient(c):
                            update_cs.add(c)
                else:
                    for c in f_der:
                        if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                            cp_cs.add(c)
                        elif isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)) and not is_static_coefficient(c):
                            update_cs.add(c)
        if isinstance(self.__functional, TimeFunctional):
            s_cp_cs = cp_cs
            s_update_cs = update_cs
            def cp_cs(s):
                cp_cs = copy.copy(s_cp_cs)
                for c in self.__functional.dependencies(s, non_symbolic = True):
                    if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                        if not isinstance(c._time_level_data[1], (int, Fraction, TimeLevel)):
                            raise DependencyException("Unexpected final time level functional dependency")
                        cp_cs.add(c)
                return cp_cs
            def update_cs(s):
                update_cs = copy.copy(s_update_cs)
                for c in self.__functional.dependencies(s, non_symbolic = True):
                    if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
                        pass
                    elif isinstance(c, (dolfin.Constant, dolfin.Function, dolfin.Expression)) and not is_static_coefficient(c):
                        update_cs.add(c)
                return update_cs

        grad = [None for i in range(len(parameters))]
        for i, parameter in enumerate(parameters):
            if self.__functional is None or isinstance(self.__functional, TimeFunctional):
                der = ufl.form.Form([])
            else:
                der = derivative(self.__functional, parameter)
            if is_empty_form(der):
                if isinstance(parameter, dolfin.Constant):
                    grad[i] = 0.0
                else:
                    assert(isinstance(parameter, dolfin.Function))
                    grad[i] = parameter.vector().copy()
                    grad[i].zero()
            else:
                grad[i] = assemble(der)

        def addto_grad(j, a_solves, a_map):
            for i, parameter in enumerate(parameters):
                for a_x in a_solves.a_x():
                    f_x = self.__a_map[a_x]
                    if not f_x in dFdm[j][i]:
                        continue
                    f_der = dFdm[j][i][f_x]
                    if isinstance(f_der, PAForm):
                        f_der_a = assemble(f_der, copy = False)
                        if isinstance(f_der_a, dolfin.GenericMatrix):
                            grad[i] -= f_der_a * a_x.vector()
                        else:
                            assert(isinstance(f_der_a, dolfin.GenericVector))
                            grad[i] -= f_der_a.inner(a_x.vector())
                    elif isinstance(f_der, ufl.core.expr.Expr):
                        f_der = evaluate_expr(f_der, copy = False)
                        if isinstance(f_der, float):
                            if isinstance(grad[i], float):
                                grad[i] += f_der * a_x.vector().sum()
                            else:
                                grad[i].axpy(f_der, a_x.vector())
                        else:
                            assert(isinstance(f_der, dolfin.GenericVector))
                            grad[i] += f_der.inner(a_x.vector())
                    else:
                        for term in f_der:
                            if isinstance(term, (ufl.constantvalue.FloatValue, dolfin.Constant)):
                                if isinstance(grad[i], float):
                                    grad[i] += float(term) * a_x.vector().sum()
                                else:
                                    grad[i].axpy(float(term), a_x.vector())
                            else:
                                assert(isinstance(term, dolfin.Function))
                                grad[i] += term.vector().inner(a_x.vector())
            return

        self.__restore_timestep_checkpoint(-1)

        self.__restore_timestep_checkpoint(-4)
        self.__forward.timestep_update(s = self.__S + 1)
        self.__adjoint.initialise()
        addto_grad(1, self.__adjoint._AdjointModel__a_initial_solves, self.__a_map)
        if not callback is None:
            callback(s = self.__S + 1)

        self.__restore_timestep_checkpoint(-3)
        for i in range(self.__S):
            self.__adjoint.update_functional(self.__S - i)
            self.__adjoint.timestep_cycle()
            self.__restore_timestep_checkpoint(self.__S - i, cp_cs = cp_cs, update_cs = update_cs)
            self.__adjoint.timestep_solve()
            addto_grad(0, self.__adjoint._AdjointModel__a_solves, self.__a_map)
            if not callback is None:
                callback(s = self.__S - i)

        self.__adjoint.final_cycle()
        self.__restore_timestep_checkpoint(-2)
        self.__adjoint.update_functional(0)
        self.__adjoint.finalise()
        addto_grad(-1, self.__adjoint._AdjointModel__a_final_solves, self.__a_map)
        if not callback is None:
            callback(s = 0)

        self.__clear_rerun_checkpoints()
        self.__restore_timestep_checkpoint(-4)

        ngrad = []
        for i, parameter in enumerate(parameters):
            if isinstance(grad[i], float):
                ngrad.append(dolfin.Constant(grad[i]))
            else:
                assert(isinstance(grad[i], dolfin.GenericVector))
                if project:
                    space = parameter.function_space()
                    ngrad.append((grad[i], dolfin.Function(space, name = "gradient_%s" % parameter.name())))
                    mass = dolfin.inner(dolfin.TestFunction(space), dolfin.TrialFunction(space)) * dolfin.dx
                    a = assembly_cache.assemble(mass)
                    linear_solver = linear_solver_cache.linear_solver(mass,
                      project_solver_parameters,
                      a = a)
                    linear_solver.set_operator(a)
                    linear_solver.solve(ngrad[-1][1].vector(), ngrad[-1][0])
                else:
                    ngrad.append(grad[i])
        return ngrad

    def taylor_test(self, parameter, ntest = 6, fact = 1.0e-4, J = None, grad = None):
        """
        Perform a Taylor remainder convergence test of a model constrained
        derivative calculation.

        Arguments:
          parameter: A static Constant or Function.
          ntest: The number of forward model runs to perform. This must be >= 2.
          fact: The relative size of the perturbation to be applied to the model
            parameter. If the parameter is a Constant then rfact is the magnitude of
            the smallest applied permutation. If the parameter is a Function then
            the smallest perturbation is a Function with a vector containing random
            values in the range [-1/2 rfact, +1/2 rfact) with a uniform probability
            distribution, where rfact = fact * || v ||_inf if || v ||_inf > 0, and
            rfact = fact otherwise, and where v is the vector associated with the
            parameter. In each forward model run the perturbation is multiplied by
            2 ** i, where i = ntest - 1, ntest - 2, ..., 1, 0.
          J: The value of the functional. Note that, if the functional is a
            TimeFunctional, the TimeFunctional is initialised with this value at the
            end of the test.
          grad: The derivative to verify. If not supplied, a derivative is computed
            using the adjoint model.

        Returns the relative orders of convergence. A successful verification will
        return orders close to 2.
        """

        if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
            raise InvalidArgumentException("parameter must be a Constant or Function")
        if not isinstance(ntest, int) or ntest <= 1:
            raise InvalidArgumentException("ntest must be an integer greater than or equal to 2")
        if not isinstance(fact, float) or fact <= 0.0:
            raise InvalidArgumentException("fact must be a positive float")
        if self.__S is None:
            raise StateException("Attempting to run Taylor remainder convergence test before finalisation")

        dolfin.info("Running Taylor remainder convergence test")

        if not grad is None:
            if isinstance(parameter, dolfin.Constant):
                if not isinstance(grad, dolfin.Constant):
                    raise InvalidArgumentException("Invalid grad type")
            else:
                assert(isinstance(parameter, dolfin.Function))
                if isinstance(grad, tuple):
                    if not len(grad) == 2 or not isinstance(grad[0], dolfin.GenericVector) or not isinstance(grad[1], dolfin.Function):
                        raise InvalidArgumentException("Invalid grad type")
                    elif not grad[0].local_size() == parameter.vector().local_size():
                        raise InvalidArgumentException("Invalid grad size")
                elif not isinstance(grad, dolfin.GenericVector):
                    raise InvalidArgumentException("Invalid grad type")
                elif not grad.local_size() == parameter.vector().local_size():
                    raise InvalidArgumentException("Invalid grad size")
        else:
            grad = self.compute_gradient(parameter)
        if isinstance(grad, dolfin.Constant):
            grad = float(grad)
        elif isinstance(grad, tuple):
            grad = grad[0]
        if not J is None:
            if not isinstance(J, float):
                raise InvalidArgumentException("J must be a float")
        else:
            J = self.compute_functional()

        if isinstance(parameter, dolfin.Constant):
            parameter_orig = float(parameter)
            perturb = fact * abs(parameter_orig)
            if perturb == 0.0:
                perturb = fact
        else:
            assert(isinstance(parameter, dolfin.Function))
            parameter_orig = dolfin.Function(parameter.function_space(), name = "%s_original" % parameter.name())
            parameter_orig.assign(parameter)
            parameter_orig_arr = parameter_orig.vector().array()
            perturb = parameter_orig.vector().copy()
            shape = parameter_orig_arr.shape
            rfact = fact * abs(parameter_orig_arr).max()
            if rfact == 0.0:
                rfact = fact
            numpy.random.seed(0)
            perturb.set_local(rfact * numpy.random.random(shape) - (0.5 * rfact))
            perturb.apply("insert")
            numpy.random.seed()

        errs_1 = []
        errs_2 = []
        for i in range(ntest - 1, -1, -1):
            if isinstance(parameter, dolfin.Constant):
                parameter.assign(dolfin.Constant(parameter_orig + (2 ** i) * perturb))

                self.reassemble_forward(parameter)
                Jp = self.compute_functional(rerun_forward = True)

                errs_1.append(abs(Jp - J))
                errs_2.append(abs(Jp - J - (2 ** i) * perturb * grad))
            else:
                assert(isinstance(parameter, dolfin.Function))
                parameter.vector()[:] = parameter_orig.vector() + (2 ** i) * perturb

                self.reassemble_forward(parameter)
                Jp = self.compute_functional(rerun_forward = True)

                errs_1.append(abs(Jp - J))
                errs_2.append(abs(Jp - J - (2 ** i) * perturb.inner(grad)))

        parameter.assign(parameter_orig)
        self.reassemble_forward(parameter)
        self.__restore_timestep_checkpoint(-4)
        if isinstance(self.__functional, TimeFunctional):
            if self.__functional.initialise.__func__.__code__.co_argcount < 2:
                self.rerun_forward()
            else:
                self.__functional.initialise(val = J)

        dolfin.info("Taylor remainder test, no adjoint errors  : %s" % str(errs_1))
        dolfin.info("Taylor remainder test, with adjoint errors: %s" % str(errs_2))

        for i, err in enumerate(errs_1):
            if err == 0.0:
                errs_1[i] = numpy.NAN
        for i, err in enumerate(errs_2):
            if err == 0.0:
                errs_2[i] = numpy.NAN
        orders_1 = numpy.empty(len(errs_1) - 1)
        for i in range(1, len(errs_1)):
            orders_1[i - 1] = -numpy.log(errs_1[i] / errs_1[i - 1]) / numpy.log(2.0)
        orders_2 = numpy.empty(len(errs_2) - 1)
        for i in range(1, len(errs_2)):
            orders_2[i - 1] = -numpy.log(errs_2[i] / errs_2[i - 1]) / numpy.log(2.0)

        if any(orders_1 < 0.9):
            dolfin.info_red("Taylor remainder test, no adjoint orders  : %s" % str(orders_1))
        else:
            dolfin.info("Taylor remainder test, no adjoint orders  : %s" % str(orders_1))
        if any(orders_2 < 1.9):
            dolfin.info_red("Taylor remainder test, with adjoint orders: %s" % str(orders_2))
        else:
            dolfin.info("Taylor remainder test, with adjoint orders: %s" % str(orders_2))

        return orders_2

    def minimise_functional(self, parameters, tolerance, parameters_0 = None,
      bounds = None, method = None, options = None, update_callback = None,
      forward_callback = None):
        """
        Minimise the registered functional subject to variations in the supplied
        static Constant s or Function s, to within the specified tolerance. This is
        a wrapper for SciPy optimisation functions.

        Arguments:
          parameters: The Constant s or Function s with respect to which the
            functional is minimised.
          tolerance: Minimisation tolerance.
          parameters_0: A list of initial guesses for the parameters. For Constant
            parameters the associated elements of parameters_0 may be None, a float,
            or a Constant. For Function parameters the associated elements of
            parameters_0 may be None, a float, or a Function. A None element
            indicates that the input value of the associated parameter is to be used
            as an initial guess.
          bounds: A list of (min, max) bounds for the parameters. The bounds have
            the same types as the elements of parameters_0. A None bound indicates
            no lower or upper bound.
          method: Optimisation method.
          options: A dictionary of additional keyword arguments to pass to the SciPy
            optimisation function.
          update_callback: Called whenever a parameter is changed.
          forward_callback: Called after each forward model run.
        """

        if isinstance(parameters, list):
            for parameter in parameters:
                if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
                    raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
        elif isinstance(parameters, (dolfin.Constant, dolfin.Function)):
            kwargs = {"method":method, "options":options, "update_callback":update_callback, "forward_callback":forward_callback}
            if bounds is None:
                kwargs["bounds"] = None
            else:
                kwargs["bounds"] = [bounds]
            if parameters_0 is None:
                kwargs["parameters_0"] = None
            else:
                kwargs["parameters_0"] = [parameters_0]
            return self.minimise_functional([parameters], tolerance, **kwargs)
        else:
            raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")

        if not parameters_0 is None:
            if not isinstance(parameters_0, list):
                raise InvalidArgumentException("parameters_0 must be a list of Nones, floats, Constant s or Function s")
            elif not len(parameters_0) == len(parameters):
                raise InvalidArgumentException("Invalid parameters_0 size")
            for i, parameter_0 in enumerate(parameters_0):
                if parameter_0 is None:
                    pass
                elif isinstance(parameters[i], dolfin.Constant):
                    if not isinstance(parameter_0, (float, dolfin.Constant)):
                        raise InvalidArgumentException("Invalid parameters_0 element type")
                else:
                    assert(isinstance(parameters[i], dolfin.Function))
                    if isinstance(parameter_0, float):
                        pass
                    elif isinstance(parameter_0, dolfin.Function):
                        if not parameter_0.vector().local_size() == parameters[i].vector().local_size():
                            raise InvalidArgumentException("Invalid parameters_0 element size")
                    else:
                        raise InvalidArgumentException("Invalid parameters_0 element type")

        if not bounds is None:
            if not isinstance(bounds, list):
                raise InvalidArgumentException("bounds must be a list of tuples of lower and upper bounds")
            elif not len(bounds) == len(parameters):
                raise InvalidArgumentException("Invalid bounds size")
            for i, bound in enumerate(bounds):
                if not isinstance(bound, tuple) or not len(bound) == 2:
                    raise InvalidArgumentException("bounds must be a list of tuples of lower and upper bounds")
                for j in range(2):
                    if bound[j] is None:
                        pass
                    elif isinstance(parameters[i], dolfin.Constant):
                        if not isinstance(bound[j], (float, dolfin.Constant)):
                            raise InvalidArgumentException("Invalid bound type")
                    else:
                        assert(isinstance(parameters[i], dolfin.Function))
                        if isinstance(bound[j], float):
                            pass
                        elif isinstance(bound[j], dolfin.Function):
                            if not bound[j].vector().local_size() == parameters[i].vector().local_size():
                                raise InvalidArgumentException("Invalid bound size")
                        else:
                            raise InvalidArgumentException("Invalid bound type")

        if not isinstance(tolerance, float) or tolerance < 0.0:
            raise InvalidArgumentException("tolerance must be a non-negative float")
        if method is None:
            if bounds is None:
                method = "BFGS"
            else:
                method = "L-BFGS-B"
        elif not isinstance(method, str):
            raise InvalidArgumentException("method must be a string")
        if not options is None and not isinstance(options, dict):
            raise InvalidArgumentException("options must be a dictionary")
        if not update_callback is None and not callable(update_callback):
            raise InvalidArgumentException("update_callback must be a callable")
        if not forward_callback is None and not callable(forward_callback):
            raise InvalidArgumentException("forward_callback must be a callable")
        if self.__functional is None:
            raise StateException("No functional defined")

        dolfin.info("Running functional minimisation")

        class Packer:
            def __init__(self, parameters):
                p = dolfin.MPI.rank(dolfin.mpi_comm_world())

                l_N = 0
                l_indices = []
                for parameter in parameters:
                    if isinstance(parameter, dolfin.Constant):
                        if p == 0:
                            n = 1
                        else:
                            n = 0
                    else:
                        n = parameter.vector().local_size()
                    l_indices.append((l_N, l_N + n))
                    l_N += n

                n_p = dolfin.MPI.size(dolfin.mpi_comm_world())
                if n_p > 1:
                    import mpi4py.MPI as MPI

                    p_N = numpy.array(MPI.COMM_WORLD.allgather(l_N), dtype = numpy.uint64)
                    g_N = p_N.sum()

                    l_arr = numpy.empty(l_N, dtype = numpy.float_)
                    g_indices = p_N[:p].sum();  g_indices = (g_indices, g_indices + l_N)

                    self.__g_N = g_N
                    self.__l_arr = l_arr
                    self.__g_indices = g_indices
                else:
                    self.__g_N = l_N
                self.__p = p
                self.__n_p = n_p
                self.__l_N = l_N
                self.__l_indices = l_indices

                return

            def serialised(self, arr):
                if self.__n_p == 1:
                    return arr.copy()
                else:
                    import mpi4py.MPI as MPI

                    p_arr = MPI.COMM_WORLD.allgather(arr)
                    g_arr = numpy.empty(self.__g_N, dtype = arr.dtype)
                    index = 0
                    for l_arr in p_arr:
                        g_arr[index:index + l_arr.shape[0]] = l_arr
                        index += l_arr.shape[0]

                    return g_arr

            def serialised_bounds(self, bounds):
                l_bounds = numpy.empty(self.__l_N, dtype = numpy.float_)
                u_bounds = numpy.empty(self.__l_N, dtype = numpy.float_)
                for i, parameter in enumerate(parameters):
                    bound = bounds[i]
                    start, end = self.__l_indices[i]
                    if isinstance(parameter, dolfin.Constant):
                        if self.__p == 0:
                            if bound[0] is None:
                                l_bounds[start] = numpy.NAN
                            else:
                                l_bounds[start] = bound[0]
                            if bound[1] is None:
                                u_bounds[start] = numpy.NAN
                            else:
                                u_bounds[start] = bound[1]
                    else:
                        if bound[0] is None:
                            l_bounds[start:end] = numpy.NAN
                        else:
                            l_bounds[start:end] = bound[0]
                        if bound[1] is None:
                            u_bounds[start:end] = numpy.NAN
                        else:
                            u_bounds[start:end] = bound[1]

                if self.__n_p > 1:
                    l_bounds = self.serialised(l_bounds)
                    u_bounds = self.serialised(u_bounds)

                bounds = []
                for i in range(self.__g_N):
                    l_bound = l_bounds[i]
                    if numpy.isnan(l_bound):
                        l_bound = None
                    u_bound = u_bounds[i]
                    if numpy.isnan(u_bound):
                        u_bound = None
                    bounds.append((l_bound, u_bound))

                return bounds

            def empty(self):
                return numpy.empty(self.__g_N, dtype = numpy.float_)

            def update(self, arr):
                if self.__n_p == 1:
                    return
                else:
                    arr[:] = self.serialised(arr[self.__g_indices[0]:self.__g_indices[1]])
                    return

            def pack(self, parameters, arr):
                if self.__n_p == 1:
                    l_arr = arr
                else:
                    l_arr = self.__l_arr
                for i, parameter in enumerate(parameters):
                    start, end = self.__l_indices[i]
                    if isinstance(parameter, dolfin.Constant):
                        if self.__p == 0:
                            l_arr[start] = float(parameter)
                    elif isinstance(parameter, dolfin.Function):
                        l_arr[start:end] = parameter.vector().array()
                    else:
                        assert(isinstance(parameter, dolfin.GenericVector))
                        l_arr[start:end] = parameter.array()
                if self.__n_p > 1:
                    arr[:] = self.serialised(l_arr)

                return

            def unpack(self, parameters, arr):
                if self.__n_p > 1:
                    arr = arr[self.__g_indices[0]:self.__g_indices[1]]
                for i, parameter in enumerate(parameters):
                    start, end = self.__l_indices[i]
                    if isinstance(parameter, dolfin.Constant):
                        if self.__p == 0:
                            val = arr[start]
                        else:
                            val = 0.0
                        parameter.assign(dolfin.MPI.sum(dolfin.mpi_comm_world(), val))
                    else:
                        assert(isinstance(parameter, dolfin.Function))
                        parameter.vector().set_local(arr[start:end])
                        parameter.vector().apply("insert")

                return
        packer = Packer(parameters)

        if not bounds is None:
            bounds = packer.serialised_bounds(bounds)

        fun_x = packer.empty()
        packer.pack(parameters, fun_x)
        rerun_forward = [False]
        reassemble_adjoint = [False]
        rerun_adjoint = [True]
        def reassemble_forward(x):
            packer.update(x)
            if not abs(fun_x - x).max() == 0.0:
                fun_x[:] = x
                packer.unpack(parameters, x)
                self.reassemble_forward(*parameters)
                rerun_forward[0] = True
                reassemble_adjoint[0] = True
                rerun_adjoint[0] = True
                if not update_callback is None:
                    update_callback()
            return

        n_fun = [0]
        def fun(x):
            reassemble_forward(x)
            if rerun_forward[0]:
                fn = self.compute_functional(rerun_forward = True, recheckpoint = True)
                rerun_forward[0] = False
                if not forward_callback is None:
                    forward_callback()
            else:
                fn = self.compute_functional()
            n_fun[0] += 1
            dolfin.info("Functional evaluation %i       : %.16e" % (n_fun[0], fn))
            if numpy.isnan(fn):
                raise StateException("NaN functional value")

            return fn

        garr = packer.empty()
        n_jac = [0]
        def jac(x):
            reassemble_forward(x)
            if rerun_forward[0]:
                dolfin.warning("Re-running forward model")
                self.rerun_forward(recheckpoint = True)
                rerun_forward[0] = False
            if reassemble_adjoint[0]:
                self.reassemble_adjoint(clear_caches = False, *parameters)
                reassemble_adjoint[0] = False
            if rerun_adjoint[0]:
                grad = self.compute_gradient(parameters)
                packer.pack(grad, garr)
                rerun_adjoint[0] = False

            n_jac[0] += 1
            grad_norm = abs(garr).max()
            dolfin.info("Gradient calculation %i inf norm: %.16e" % (n_jac[0], grad_norm))

            return garr

        if not parameters_0 is None:
            for i, parameter_0 in enumerate(parameters_0):
                if parameter_0 is None:
                    pass
                elif isinstance(parameters[i], dolfin.Function) and isinstance(parameter_0, float):
                    parameters[i].vector()[:] = parameter_0
                else:
                    parameters[i].assign(parameter_0)
        x = packer.empty()
        packer.pack(parameters, x)

        res = scipy.optimize.minimize(fun, x, method = method, jac = jac, bounds = bounds, tol = tolerance, options = options)
        if not res["success"]:
            raise StateException("scipy.optimize.minimize failure")
        dolfin.info("scipy.optimize.minimize success with %i functional evaluation(s)" % res["nfev"])
        x = res["x"]

        reassemble_forward(x)
        if rerun_forward[0]:
            dolfin.warning("Re-running forward model")
            self.rerun_forward(recheckpoint = True)
        if reassemble_adjoint[0]:
            self.reassemble_adjoint(clear_caches = False, *parameters)

        dolfin.info("Functional minimisation complete after %i functional evaluation(s) and %i gradient calculation(s)" % (n_fun[0], n_jac[0]))

        return res
