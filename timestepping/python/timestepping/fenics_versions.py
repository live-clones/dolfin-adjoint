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
import ffc
import instant
import numpy
import ufl

from .exceptions import *

__all__ = \
  [
    "Version",
    "dolfin_version",
    "ffc_version",
    "instant_version",
    "ufl_version"
  ]

class Version(object):
    """
    Defines and enables comparisons between version numbers. Is supplied with
    rich comparison methods.

    Constructor arguments:
      ver: One of:
          1. A non-negative integer.
        or:
          2. A period delimited string. If the string ends in "-dev" or "+" then
             this ending is treated as ".1".
        or:
          3. An object that can be cast to an integer numpy array with
             non-negative elements.
    """

    def __init__(self, ver):
        if isinstance(ver, int):
            ver = [ver]
        elif isinstance(ver, str):
            if ver.endswith("-dev"):
                ver = ver[:-4].split(".") + [1]
            elif ver.endswith("+"):
                ver = ver[:-1].split(".") + [1]
            else:
                ver = ver.split(".")
        ver = numpy.array(ver, dtype = numpy.int)
        if len(ver) == 0 or not (ver >= 0).all():
            raise InvalidArgumentException("Invalid version")
        self.__ver = ver
        return

    def tuple(self):
        """
        Return a tuple representation of the version number.
        """

        return tuple(self.__ver)

    def __len__(self):
        return self.__ver.shape[0]

    def __str__(self):
        s = str(self.__ver[0])
        for i in range(1, len(self.__ver)):
            s += ".%i" % self.__ver[i]
        return s

    def __eq__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        n = min(len(self.__ver), len(other.__ver))
        for i in range(n):
            if not self.__ver[i] == other.__ver[i]:
                return False
        for i in range(n, len(self.__ver)):
            if not self.__ver[i] == 0:
                return False
        for i in range(n, len(other.__ver)):
            if not other.__ver[i] == 0:
                return False
        return True

    def __gt__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        n = min(len(self.__ver), len(other.__ver))
        for i in range(n):
            if other.__ver[i] > self.__ver[i]:
                return False
        for i in range(n, len(self.__ver)):
            if self.__ver[i] > 0:
                return True
        for i in range(n, len(other.__ver)):
            if other.__ver[i] > 0:
                return False
        return False

    def __lt__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        n = min(len(self.__ver), len(other.__ver))
        for i in range(n):
            if self.__ver[i] < other.__ver[i]:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def __cmp__(self, other):
        if self > other:
            return 1
        elif self < other:
            return -1
        else:
            return 0

def dolfin_version():
    """
    Return the current DOLFIN version, as a Version.
    """

    return Version(dolfin.__version__)

def ffc_version():
    """
    Return the current FFC version, as a Version.
    """

    return Version(ffc.__version__)

def instant_version():
    """
    Return the current Instant version, as a Version.
    """

    return Version(instant.__version__)

def ufl_version():
    """
    Return the current UFL version, as a Version.
    """

    return Version(ufl.__version__)
