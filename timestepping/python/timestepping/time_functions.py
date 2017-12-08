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
from fractions import Fraction

import dolfin
import ufl

from .exceptions import *
from .fenics_overrides import *
from .time_levels import *
from .versions import *

__all__ = \
  [
    "AdjointTimeFunction",
    "TimeFunction",
    "WrappedFunction"
  ]

class WrappedFunction(dolfin.Function):
    """
    Wraps dolfin Function objects to enable Function aliasing, deferred
    allocation, and Function deallocation. Always has a name and a FunctionSpace.

    Constructor arguments:
      arg: One of:
          1. A FunctionSpace. The WrappedFunction is assigned the given function
             space, but is not associated with any DOLFIN Function.
        or:
          2. A Function. The WrappedFunction is assigned the function space of
             the given Function, and wraps the Function.
      name: A string defining the name of the function.
      label: A string defining the label of the function.
    """

    def __init__(self, arg, name = "u", label = "a WrappedFunction"):
        if not isinstance(name, str):
            raise InvalidArgumentException("name must be a string")
        if not isinstance(label, str):
            raise InvalidArgumentException("label must be a string")
        self.__fn = None
        if isinstance(arg, dolfin.cpp.function.FunctionSpace):
            self.__space = arg
                                                                                   # Work around DOLFIN id issues
            ufl.coefficient.Coefficient.__init__(self, self.__space.ufl_element(), count = dolfin.Constant(0).id())
        elif isinstance(arg, dolfin.Function):
            self.__space = arg.function_space()
                                                                                   # Work around DOLFIN id issues
            ufl.coefficient.Coefficient.__init__(self, self.__space.ufl_element(), count = dolfin.Constant(0).id())
            self.wrap(arg)
        else:
            raise InvalidArgumentException("Require FunctionSpace or Function as first argument")
        self.__name = name
        self.__label = label

        return

    def allocate(self):
        """
        Wrap a newly allocated Function.
        """

        self.wrap(dolfin.Function(self.__space, name = self.__name, label = self.__label))

        return

    def deallocate(self):
        """
        Alias to the unwrap method.
        """

        self.unwrap()

        return

    def wrap(self, fn):
        """
        Wrap the supplied Function.
        """

        if not isinstance(fn, (dolfin.Function, WrappedFunction)):
            raise InvalidArgumentException("fn must be a Function or WrappedFunction")
        # This comparison is very expensive. Trust that the caller knows what it's
        # doing.
#    elif not fn.function_space() == self.__space:
#      raise InvalidArgumentException("Invalid FunctionSpace")

        if isinstance(fn, WrappedFunction):
            fn = fn.__fn

        self.unwrap()
        self.__fn = fn
        self.this = fn.this

        return

    def unwrap(self):
        """
        Unwrap, so that the WrappedFunction no longer wraps any DOLFIN Function.
        """

        if not self.__fn is None:
            del(self.this)
            self.__fn = None

        return

    def is_wrapping(self):
        """
        Return whether the WrappedFunction is currently wrapping any DOLFIN
        Function.
        """

        return not self.__fn is None

    def fn(self):
        """
        Return the currently wrapped function, as a Function.
        """

        return self.__fn

    def function_space(self):
        """
        Return the function space, as a FunctionSpace.
        """

        return self.__space

    def name(self):
        """
        Return the function name, as a string.
        """

        return self.__name

    def label(self):
        """
        Return the function label, as a string.
        """

        return self.__label

    def rename(self, name, label):
        """
        Rename the WrappedFunction.
        """

        if not isinstance(name, str):
            raise InvalidArgumentException("name must be a string")

        self.__name = name
        self.__label = label

        return

class TimeFunction(TimeLevels):
    """
    A function defined on a number of time levels. Individual Function s can
    be accessed by indexing directly into the object.

    Constructor arguments:
      tlevels: A TimeLevels prescribing the time levels on which the function is
        defined.
      space: The FunctionSpace on which the function is defined.
      name: A string defining the name of the function.
    """

    def __init__(self, tlevels, space, name = "u"):
        if not isinstance(tlevels, TimeLevels):
            raise InvalidArgumentException("tlevels must be a TimeLevels")
        if not isinstance(space, dolfin.cpp.function.FunctionSpace):
            raise InvalidArgumentException("space must be a FunctionSpace")
        if not isinstance(name, str):
            raise InvalidArgumentException("name must be a string")

        fns = {}
        lfns = {}
        for level in tlevels.levels():
            fns[level] = WrappedFunction(dolfin.Function(space, name = "%s_%s" % (name, level)), name = "%s_%s" % (name, level))
            fns[level]._time_level_data = (self, level)

            nlevel = N + level.offset()
            lfns[nlevel] = WrappedFunction(space, name = "%s_%s" % (name, nlevel))
            lfns[nlevel]._time_level_data = (self, nlevel)

        self._TimeLevels__copy_time_levels(tlevels)
        self.__name = name
        self.__fns = fns
        self.__space = space
        self.__lfns = lfns
        self.__id = fns[tlevels.levels()[0]].id()

        return

    def __getitem__(self, key):
        if isinstance(key, (int, Fraction)):
            if not key in self._TimeLevels__offsets:
                raise InvalidArgumentException("key out of range")
            if not key in self.__lfns:
                self.__lfns[key] = WrappedFunction(self.__fns[n + key], name = "%s_%s" % (self.__name, key))
                self.__lfns[key]._time_level_data = (self, key)
            return self.__lfns[key]
        elif isinstance(key, TimeLevel):
            return self.__fns[key]
        elif isinstance(key, FinalTimeLevel):
            return self.__lfns[key]
        else:
            raise InvalidArgumentException("key must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

    def __eq__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return self.__id == other.__id

    def __gt__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return self.__id > other.__id

    def __lt__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return self.__id < other.__id

    def __ne__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return not self == other

    def __ge__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return not self < other

    def __le__(self, other):
        if not isinstance(other, TimeFunction):
            raise InvalidArgumentException("other must be a TimeFunction")
        return not self > other

    def __hash__(self):
        return hash(self.__id)

    def name(self):
        """
        Return the name of the TimeFunction, as a string.
        """

        return self.__name

    def function_space(self):
        """
        Return the function space of the TimeFunction, as a FunctionSpace.
        """

        return self.__space

    def all_levels(self):
        """
        Return all levels on which the TimeFunction is defined, as a list of
        integers, Fraction s, TimeLevel s or FinalTimeLevel s.
        """

        return list(self._TimeLevels__levels) + list(self.__lfns.keys())

    def has_level(self, level):
        """
        Return whether the TimeFunction is defined on the specified level. level
        may be an integer, Fraction, TimeLevel or FinalTimeLevel.
        """

        if not isinstance(level, (int, Fraction, TimeLevel, FinalTimeLevel)):
            raise InvalidArgumentException("level must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

        if isinstance(level, TimeLevel):
            return level in self.__fns
        else:
            return level in self.__lfns

    def fns(self):
        """
        Return all Function s associated with the TimeFunction.
        """

        return list(self.__fns.values())

    def initial_levels(self):
        """
        Return the initial time levels on which the TimeFunction is defined, as a
        list of integers or Fraction s.
        """

        levels = []
        for level in self.__lfns:
            if isinstance(level, (int, Fraction)):
                levels.append(level)

        return levels

    def final_levels(self):
        """
        Return the final time levels on which the TimeFunction is defined, as a list
        of FinalTimeLevel s.
        """

        levels = []
        for level in self.__lfns:
            if isinstance(level, FinalTimeLevel):
                levels.append(level)

        return levels

    def initial_cycle_map(self):
        """
        Return the initial cycle map, as an OrderedDict with Function keys and
        values.
        """

        cycle_map = OrderedDict()
        for level in self.levels():
            if level.offset() in self.__lfns:
                cycle_map[level] = level.offset()

        return cycle_map

    def final_cycle_map(self):
        """
        Return the final cycle map, as an OrderedDict with Function keys and values.
        """

        cycle_map = OrderedDict()
        for level in self.levels():
            nlevel = N + level.offset()
            if nlevel in self.__lfns:
                cycle_map[nlevel] = level

        return cycle_map

    def initial_cycle(self):
        """
        Perform the initial cycle. After the initial cycle the Function s on
        TimeLevel s are well-defined, but those on initial levels contain
        arbitrary data.
        """

        cycle_map = self.initial_cycle_map()
        for level in cycle_map:
            self.__fns[level].wrap(self.__lfns[cycle_map[level]])

        for level in self.levels():
            if not level in cycle_map:
                self.__fns[level].allocate()
                self.__fns[level].vector().zero()

        return

    def cycle(self, extended = True):
        """
        Perform a timestep cycle. If extended is true, use the extended cycle map
        to perform the cycle, via aliasing. Otherwise, use the cycle map to
        perform the cycle, via copying.
        """

        if extended:
            fns = {}
            for level in self.levels():
                fns[level] = self.__fns[level].fn()

            cycle_map = self.extended_cycle_map()
            for level in cycle_map:
                self.__fns[level].wrap(fns[cycle_map[level]])
        else:
            cycle_map = self.cycle_map()
            for level in cycle_map:
                self.__fns[level].assign(self.__fns[cycle_map[level]])

        return

    def final_cycle(self):
        """
        Perform the final cycle. After the final cycle the Function s on
        FinalTimeLevel s are well-defined, but those on TimeLevel s contain
        arbitrary data.
        """

        cycle_map = self.final_cycle_map()
        for level in cycle_map:
            self.__lfns[level].wrap(self.__fns[cycle_map[level]])

        return

class AdjointTimeFunction(TimeLevels):
    """
    An adjoint function defined on a number of time levels.

    Constructor arguments:
      tfn: The associated forward TimeFunction.
    """

    def __init__(self, tfn):
        if not isinstance(tfn, TimeFunction):
            raise InvalidArgumentException("tfn must be a TimeFunction")

        name = tfn.name()

        fns = {}
        for level in tfn.levels():
            fns[level] = dolfin.Function(name = "%s_%s_adjoint" % (name, level), *[tfn.function_space()])
            fns[level]._time_level_data = (self, level)
            fns[level]._adjoint_data = [tfn[level]]
        for level in tfn.initial_levels():
            fns[level] = WrappedFunction(fns[n + level], name = "%s_%s_adjoint" % (name, level))
            fns[level]._time_level_data = (self, level)
            fns[level]._adjoint_data = [tfn[level]]
        for level in tfn.final_levels():
            fns[level] = WrappedFunction(fns[n + level.offset()], name = "%s_%s_adjoint" % (name, level))
            fns[level]._time_level_data = (self, level)
            fns[level]._adjoint_data = [tfn[level]]

        self._TimeLevels__copy_time_levels(tfn)
        self.__name = name
        self.__fns = fns
        self.__space = tfn.function_space()
        self.__tfn = tfn

        return

    def __getitem__(self, key):
        if isinstance(key, (int, Fraction, TimeLevel, FinalTimeLevel)):
            return self.__fns[key]
        else:
            raise InvalidArgumentException("key must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

    def name(self):
        """
        Return the name of the AdjointTimeFunction, as a string.
        """

        return self.__name

    def has_level(self, level):
        """
        Return whether the AdjointTimeFunction is defined on the specified level.
        level may be an integer, Fraction, TimeLevel or FinalTimeLevel.
        """

        if not isinstance(level, (int, Fraction, TimeLevel, FinalTimeLevel)):
            raise InvalidArgumentException("level must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

        return level in self.__fns

    def forward(self):
        """
        Return the forward TimeFunction associated with the AdjointTimeFunction.
        """

        return self.__tfn

    def zero(self):
        """
        Zero the TimeFunction.
        """

        for level in self.levels():
            self[level].vector().zero()

        return
