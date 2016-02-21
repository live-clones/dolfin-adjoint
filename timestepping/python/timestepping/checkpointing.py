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
import cPickle
import copy
import os

import dolfin

from exceptions import *

__all__ = \
  [
    "Checkpointer",
    "DiskCheckpointer",
    "MemoryCheckpointer"
  ]

class Checkpointer(object):
    """
    A template for Constant and Function storage.
    """

    def __init__(self):
        return

    def __pack(self, c):
        if isinstance(c, dolfin.Constant):
            return float(c)
        else:
            assert(isinstance(c, dolfin.Function))
            return c.vector().array()

    def __unpack(self, c, c_c):
        if isinstance(c, dolfin.Constant):
            c.assign(c_c)
        else:
            assert(isinstance(c, dolfin.Function))
            c.vector().set_local(c_c)
            c.vector().apply("insert")

        return

    def __verify(self, c, c_c, tolerance = 0.0):
        if isinstance(c, dolfin.Constant):
            err = abs(float(c) - c_c)
            if err > tolerance:
                raise CheckpointException("Invalid checkpoint data for Constant with value %.6g, error %.6g" % (float(c), err))
        else:
            assert(isinstance(c, dolfin.Function))
            assert(c_c.shape[0] == c.vector().local_size())
            if c_c.shape[0] == 0:
                err = 0.0
            else:
                err = abs(c.vector().array() - c_c).max()
            if err > tolerance:
                raise CheckpointException("Invalid checkpoint data for Function %s, error %.6g" % (c.name(), err))

        return

    def __check_cs(self, cs):
        if not isinstance(cs, (list, set)):
            raise InvalidArgumentException("cs must be a list of Constant s or Function s")
        for c in cs:
            if not isinstance(c, (dolfin.Constant, dolfin.Function)):
                raise InvalidArgumentException("cs must be a list of Constant s or Function s")

        if not isinstance(cs, set):
            cs = set(cs)
        cs = list(cs)
        def cmp(x, y):
            return x.id() - y.id()
        cs.sort(cmp = cmp)

        return cs

    def checkpoint(self, key, cs):
        """
        Store, with the supplied key, the supplied Constant s and Function s.
        """

        raise AbstractMethodException("checkpoint method not overridden")

    def restore(self, key, cs = None):
        """
        Restore Constant s and Function s with the given key. If cs is supplied,
        only restore Constant s and Function s found in cs.
        """

        raise AbstractMethodException("restore method not overridden")

    def has_key(key):
        """
        Return whether any data is associated with the given key.
        """

        raise AbstractMethodException("has_key method not overridden")

    def verify(self, key, tolerance = 0.0):
        """
        Verify data associated with the given key, with the specified tolerance.
        """

        raise AbstractMethodException("verify method not overridden")

    def remove(self, key):
        """
        Remove data associated with the given key.
        """

        raise AbstractMethodException("remove method not overridden")

    def clear(self, keep = []):
        """
        Clear all stored data, except for those with keys in keep.
        """

        raise AbstractMethodException("clear method not overridden")

class MemoryCheckpointer(Checkpointer):
    """
    Constant and Function storage in memory.
    """

    def __init__(self):
        Checkpointer.__init__(self)

        self.__cache = {}

        return

    def checkpoint(self, key, cs):
        """
        Store, with the supplied key, the supplied Constant s and Function s.
        """

        if key in self.__cache:
            raise CheckpointException("Attempting to overwrite checkpoint with key %s" % str(key))
        cs = self._Checkpointer__check_cs(cs)

        c_cs = OrderedDict()
        for c in cs:
            c_cs[c] = self._Checkpointer__pack(c)

        self.__cache[key] = c_cs

        return

    def restore(self, key, cs = None):
        """
        Restore Constant s and Function s with the given key. If cs is supplied,
        only restore Constant s and Function s found in cs.
        """

        if not key in self.__cache:
            raise CheckpointException("Missing checkpoint with key %s" % str(key))
        if not cs is None:
            cs = self._Checkpointer__check_cs(cs)

        c_cs = self.__cache[key]
        if cs is None:
            cs = c_cs.keys()

        for c in cs:
            self._Checkpointer__unpack(c, c_cs[c])

        return

    def has_key(self, key):
        """
        Return whether any data is associated with the given key.
        """

        return key in self.__cache

    def verify(self, key, tolerance = 0.0):
        """
        Verify data associated with the given key, with the specified tolerance.
        """

        if not key in self.__cache:
            raise CheckpointException("Missing checkpoint with key %s" % str(key))
        if not isinstance(tolerance, float) or tolerance < 0.0:
            raise InvalidArgumentException("tolerance must be a non-negative float")
        c_cs = self.__cache[key]

        try:
            for c in c_cs:
                self._Checkpointer__verify(c, c_cs[c], tolerance = tolerance)
            dolfin.info("Verified checkpoint with key %s" % str(key))
        except CheckpointException as e:
            dolfin.info(str(e))
            raise CheckpointException("Failed to verify checkpoint with key %s" % str(key))

        return

    def remove(self, key):
        """
        Remove data associated with the given key.
        """

        if not key in self.__cache:
            raise CheckpointException("Missing checkpoint with key %s" % str(key))

        del(self.__cache[key])

        return

    def clear(self, keep = []):
        """
        Clear all stored data, except for those with keys in keep.
        """

        if not isinstance(keep, list):
            raise InvalidArgumentException("keep must be a list")

        if len(keep) == 0:
            self.__cache = {}
        else:
            for key in copy.copy(self.__cache.keys()):
                if not key in keep:
                    del(self.__cache[key])

        return

class DiskCheckpointer(Checkpointer):
    """
    Constant and Function storage on disk. All keys handled by a DiskCheckpointer
    are internally cast to strings.

    Constructor arguments:
      dirname: The directory in which data is to be stored.
    """

    def __init__(self, dirname = "checkpoints~"):
        if not isinstance(dirname, str):
            raise InvalidArgumentException("dirname must be a string")

        Checkpointer.__init__(self)

        if dolfin.MPI.process_number() == 0:
            if not os.path.exists(dirname):
                os.mkdir(dirname)
        dolfin.MPI.barrier()

        self.__dirname = dirname
        self.__filenames = {}
        self.__id_map = {}

        return

    def __filename(self, key):
        return os.path.join(self.__dirname, "checkpoint_%s_%i" % (str(key), dolfin.MPI.process_number()))

    def checkpoint(self, key, cs):
        """
        Store, with the supplied key, the supplied Constant s and Function s. The
        key is internally cast to a string.
        """

        key = str(key)
        if key in self.__filenames:
            raise CheckpointException("Attempting to overwrite checkpoint with key %s" % key)
        cs = self._Checkpointer__check_cs(cs)

        c_cs = OrderedDict()
        id_map = {}
        for c in cs:
            c_id = c.id()
            c_cs[c_id] = self._Checkpointer__pack(c)
            id_map[c_id] = c

        filename = self.__filename(key)
        handle = open(filename, "wb")
        pickler = cPickle.Pickler(handle, -1)
        pickler.dump(c_cs)

        self.__filenames[key] = filename
        self.__id_map[key] = id_map

        return

    def restore(self, key, cs = None):
        """
        Restore Constant s and Function s with the given key. If cs is supplied,
        only restore Constant s and Function s found in cs. The key is internally
        cast to a string.
        """

        key = str(key)
        if not key in self.__filenames:
            raise CheckpointException("Missing checkpoint with key %s" % key)
        if not cs is None:
            cs = self._Checkpointer__check_cs(cs)
            cs = [c.id() for c in cs]

        handle = open(self.__filename(key), "rb")
        pickler = cPickle.Unpickler(handle)
        c_cs = pickler.load()
        if cs is None:
            cs = c_cs.keys()

        id_map = self.__id_map[key]
        for c_id in cs:
            c = id_map[c_id]
            self._Checkpointer__unpack(c, c_cs[c_id])

        return

    def has_key(self, key):
        """
        Return whether any data is associated with the given key. The key is
        internally cast to a string.
        """

        key = str(key)
        return key in self.__filenames

    def verify(self, key, tolerance = 0.0):
        """
        Verify data associated with the given key, with the specified tolerance. The
        key is internally cast to a string.
        """

        key = str(key)
        if not key in self.__filenames:
            raise CheckpointException("Missing checkpoint with key %s" % key)
        if not isinstance(tolerance, float) or tolerance < 0.0:
            raise InvalidArgumentException("tolerance must be a non-negative float")
        handle = open(self.__filename(key), "rb")
        pickler = cPickle.Unpickler(handle)
        c_cs = pickler.load()

        try:
            id_map = self.__id_map[key]
            for c_id in c_cs:
                c = id_map[c_id]
                self._Checkpointer__verify(c, c_cs[c_id], tolerance = tolerance)
            dolfin.info("Verified checkpoint with key %s" % key)
        except CheckpointException as e:
            dolfin.info(str(e))
            raise CheckpointException("Failed to verify checkpoint with key %s" % key)

        return

    def remove(self, key):
        """
        Remove data associated with the given key. The key is internally cast to a
        string.
        """

        key = str(key)
        if not key in self.__filenames:
            raise CheckpointException("Missing checkpoint with key %s" % key)

#    os.remove(self.__filenames[key])
        del(self.__filenames[key])
        del(self.__id_map[key])

        return

    def clear(self, keep = []):
        """
        Clear all stored data, except for those with keys in keep. The keys are
        internally cast to strings.
        """

        if not isinstance(keep, list):
            raise InvalidArgumentException("keep must be a list")

        if len(keep) == 0:
#      for key in self.__filenames:
#        os.remove(self.__filenames[key])
            self.__filenames = {}
            self.__id_map = {}
        else:
            keep = [str(key) for key in keep]
            for key in copy.copy(self.__filenames.keys()):
                if not key in keep:
         #         os.remove(self.__filenames[key])
                    del(self.__filenames[key])
                    del(self.__id_map[key])

        return
