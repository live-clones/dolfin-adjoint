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
from fractions import Fraction

from .exceptions import *

__all__ = \
  [
    "FinalTimeLevel",
    "N",
    "TimeLevel",
    "TimeLevels",
    "n"
  ]

class TimeLevel(object):
    """
    A single model time level, with a specified logical offset. Is supplied with
    rich comparison methods, which compares the offsets of two TimeLevel s.
    Addition or subtraction of an integer or Fraction is also well defined, and
    returns a TimeLevel with the associated offset increased or decreased by the
    given integer or Fraction.

    Constructor arguments:
      arg: One of:
          1. No arguments. The time level is assigned an offset of zero.
        or:
          2. An integer or Fraction, which is used to set the offset.
        or:
          3. A TimeLevel, acting as a copy constructor.
    """

    def __init__(self, arg = 0):
        if isinstance(arg, (int, Fraction)):
            offset = arg
        elif isinstance(arg, TimeLevel):
            offset = arg.__offset
        else:
            raise InvalidArgumentException("Require no arguments, or TimeLevel, integer, or Fraction")

        self.__offset = offset

        return

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            return TimeLevel(self.__offset + other)
        else:
            raise InvalidArgumentException("other must be an integer or Fraction")

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            return TimeLevel(self.__offset - other)
        else:
            raise InvalidArgumentException("other must be an integer or Fraction")

    def __eq__(self, other):
        if isinstance(other, TimeLevel):
            return self.__offset == other.__offset
        else:
            raise InvalidArgumentException("other must be a TimeLevel")

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, TimeLevel):
            return self.__offset < other.__offset
        else:
            raise InvalidArgumentException("other must be a TimeLevel")

    def __le__(self, other):
        if isinstance(other, TimeLevel):
            return self.__offset <= other.__offset
        else:
            raise InvalidArgumentException("other must be a TimeLevel")

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __cmp__(self, other):
        if self > other:
            return 1
        elif self < other:
            return -1
        else:
            return 0

    def __hash__(self):
        return hash((0, self.__offset))

    def __str__(self):
        if self.__offset == 0:
            return "n"
        elif self.__offset > 0:
            return "n+%s" % self.__offset
        else:
            return "n-%s" % -self.__offset

    def offset(self):
        """
        Return the offset associated with the TimeLevel.
        """

        return self.__offset

class FinalTimeLevel(object):
    """
    A single final model time level, with a specified logical offset. Is supplied
    with rich comparison methods, which compares the offsets of two
    FinalTimeLevel s. Addition or subtraction of an integer or Fraction is also
    well defined, and returns a FinalTimeLevel with the associated offset
    increased or decreased by the given integer or Fraction.

    Constructor arguments:
      arg: One of:
          1. No arguments. The final time level is assigned an offset of zero.
        or:
          2. An integer or Fraction, which is used to set the offset.
        or:
          3. A FinalTimeLevel, acting as a copy constructor.
    """

    def __init__(self, arg = 0):
        if isinstance(arg, (int, Fraction)):
            offset = arg
        elif isinstance(arg, FinalTimeLevel):
            offset = arg.__offset
        else:
            raise InvalidArgumentException("Require no arguments, or FinalTimeLevel, integer, or Fraction")

        self.__offset = offset

        return

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            return FinalTimeLevel(self.__offset + other)
        else:
            raise InvalidArgumentException("other must be an integer or Fraction")

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            return FinalTimeLevel(self.__offset - other)
        else:
            raise InvalidArgumentException("other must be an integer or Fraction")

    def __eq__(self, other):
        if isinstance(other, FinalTimeLevel):
            return self.__offset == other.__offset
        else:
            raise InvalidArgumentException("other must be a FinalTimeLevel")

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, FinalTimeLevel):
            return self.__offset < other.__offset
        else:
            raise InvalidArgumentException("other must be a FinalTimeLevel")

    def __le__(self, other):
        if isinstance(other, FinalTimeLevel):
            return self.__offset <= other.__offset
        else:
            raise InvalidArgumentException("other must be a FinalTimeLevel")

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return self < other

    def __cmp__(self, other):
        if self > other:
            return 1
        elif self < other:
            return -1
        else:
            return 0

    def __hash__(self):
        return hash((1, self.__offset))

    def __str__(self):
        if self.__offset == 0:
            return "N"
        elif self.__offset > 0:
            return "N+%s" % self.__offset
        else:
            return "N-%s" % -self.__offset

    def offset(self):
        """
        Return the offset associated with the FinalTimeLevel.
        """

        return self.__offset

# Logical handles denoting an arbitrary time level "n" and the final time level
# "N".
n = TimeLevel()
N = FinalTimeLevel()

class TimeLevels(object):
    """
    A unique set of time levels and a timestep variable cycle map. The cycle map
    defines the cycling of data at the end of a model time step.

    Constructor arguments:
      levels: A unique list of TimeLevel s.
      cycle_map: A dictionary, defining a one-to-one map from "past" levels to
        "later" levels. At the end of a model timestep the keys of the
        dictionary are to be replaced with the data associated with the values of
        the dictionary.
      last_past_level: A TimeLevel used to define the "present" point.
        Levels <= this time level are treated as being in the past, while levels
        > this time level are treated as being in the future.
    """

    def __init__(self, levels, cycle_map, last_past_level = n):
        if not isinstance(last_past_level, TimeLevel):
            raise InvalidArgumentException("last_past_level must be a TimeLevel")
        if not isinstance(levels, list):
            raise InvalidArgumentException("levels must be a list of TimeLevel s")
        for level in levels:
            if not isinstance(level, TimeLevel):
                raise InvalidArgumentException("levels must be a list of TimeLevel s")
        if len(levels) == 0:
            raise InvalidArgumentException("levels must include at least one TimeLevel")
        levels = tuple(sorted(list(set(levels))))

        if not isinstance(cycle_map, dict):
            raise InvalidArgumentException("cycle_map must be a dictionary with TimeLevel keys and values")
        for level in cycle_map:
            nlevel = cycle_map[level]
            if not isinstance(level, TimeLevel) or not isinstance(nlevel, TimeLevel):
                raise InvalidArgumentException("cycle_map must be a dictionary with TimeLevel keys and values")
            elif not level in levels or not nlevel in levels:
                raise InvalidArgumentException("Level in cycle_map not in levels")
            elif nlevel <= level:
                raise InvalidArgumentException("Must map earlier to later levels")
            elif level > last_past_level:
                raise InvalidArgumentException("Cannot map future to future levels")

        if not len(set(cycle_map.values())) == len(list(cycle_map.values())):
            raise InvalidArgumentException("cycle_map must be one-to-one")

        cm_levels = sorted(cycle_map.keys())
        ncycle_map = OrderedDict()
        for level in cm_levels:
            ncycle_map[level] = cycle_map[level]
        cycle_map = ncycle_map

        key_levels = list(cycle_map.keys())
        for level in cycle_map.values():
            if level in key_levels:
                key_levels.remove(level)
        val_levels = list(cycle_map.values())
        for level in cycle_map.keys():
            if level in val_levels:
                val_levels.remove(level)
        assert(len(key_levels) == len(val_levels))
        ext_cycle_map = copy.copy(cycle_map)
        for key, value in zip(val_levels, key_levels):
            ext_cycle_map[key] = value

        self.__levels = levels
        self.__offsets = [level.offset() for level in levels]
        self.__cycle_map = cycle_map
        self.__ext_cycle_map = ext_cycle_map
        self.__last_past_level = last_past_level

        return

    def __copy_time_levels(self, other):
        self.__levels = other.__levels
        self.__offsets = other.__offsets
        self.__cycle_map = other.__cycle_map
        self.__ext_cycle_map = other.__ext_cycle_map
        self.__last_past_level = other.__last_past_level

        return

    def levels(self):
        """
        Return the levels, as a tuple of TimeLevel s.
        """

        return self.__levels

    def cycle_map(self):
        """
        Return the cycle map, as an OrderedDict with TimeLevel keys and values.
        """

        return self.__cycle_map

    def extended_cycle_map(self):
        """
        Return the extended cycle map, as an OrderedDict with TimeLevel keys and
        values. This maps replaced past time levels to non-replaced time levels. For
        example, with:
          levels = [n, n + 1], cycle_map = {n:n + 1}
        then the extended cycle map is an OrderedDict with keys and values
        equivalent to:
          extended_cycle_map = {n:n + 1, n + 1:n}
        """

        return self.__ext_cycle_map

    def last_past_level(self):
        """
        Return the last past level, as a TimeLevel.
        """

        return self.__last_past_level
