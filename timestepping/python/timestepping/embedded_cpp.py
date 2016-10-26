#!/usr/bin/env python2

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2016 University of Edinburgh
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
import ctypes
import os

import dolfin
import instant
import numpy

from exceptions import *

__all__ = \
  [
    "EmbeddedCpp",
    "CellKernel",
    "double_arr",
    "int_arr",
    "long_arr"
  ]

int_arr, long_arr, double_arr = 2, 3, 4

class EmbeddedCpp(object):
    """
    A wrapper for short sections of embedded C++ code.

    Constructor arguments:
      code:          C++ code.
      includes:      Code which can, for example be used to include header files.
      include_dirs:  Header file directories.
    Remaining keyword arguments form a list of name:type pairs, with:
      name:          The name of a variable in the code, which will be passed from
                     Python.
      type:          One of int, float, int_arr, long_arr, double_arr,
                     DirichletBC, Function, GenericMatrix, GenericVector, or Mesh,
                     identifying the variable type.
    """

    __boost_classes = {dolfin.cpp.DirichletBC:dolfin.DirichletBC,
                       dolfin.DirichletBC:"DirichletBC",
                       dolfin.Function:"Function",
                       dolfin.GenericMatrix:"GenericMatrix",
                       dolfin.GenericVector:"GenericVector",
                       dolfin.Mesh:"Mesh"}

    __default_includes = """#include "dolfin.h" """

    def __init__(self, code, includes = "", include_dirs = [], **kwargs):
        if not isinstance(code, str):
            raise InvalidArgumentException("code must be a string")
        if not isinstance(includes, str):
            raise InvalidArgumentException("includes must be a string")
        if not isinstance(include_dirs, list):
            raise InvalidArgumentException("include_dirs must be a list of strings")
        for path in include_dirs:
            if not isinstance(path, str):
                raise InvalidArgumentException("include_dirs must be a list of strings")
        for arg in kwargs.keys():
            if not isinstance(arg, str):
                raise InvalidArgumentException("Argument name must be a string")
        for arg in kwargs.values():
            if not arg in [int, float, int_arr, double_arr, long_arr] + self.__boost_classes.keys():
                raise InvalidArgumentException("Argument type must be int, float, int_arr, long_arr, double_arr, DirichletBC, Function, GenericMatrix, GenericVector or Mesh")

        args = copy.copy(kwargs)
        for arg in args:
            if arg in self.__boost_classes.keys():
                cls = self.__boost_classes[arg]
                while not isinstance(cls, str):
                    cls = self.__boost_classes[cls]
                args[arg] = cls

        includes = \
    """
    #include <memory>
    using namespace std;

    %s""" % includes

        self.__code = code
        self.__includes = \
    """
    %s

    %s""" % (includes, self.__default_includes)
        self.__include_dirs = copy.copy(include_dirs)
        self.__args = args
        self.__lib = None

        return

    def compile(self):
        """
        Compile the code.
        """

        args = ""
        argtypes = []
        cast_code = ""
        for name in sorted(self.__args.keys()):
            arg = self.__args[name]
            if len(args) > 0:
                args += ", "
            if arg == int:
                args += "int %s" % name
                argtypes.append(ctypes.c_int)
            elif arg == float:
                args += "double %s" % name
                argtypes.append(ctypes.c_double)
            elif arg == int_arr:
                args += "int* %s" % name
                argtypes.append(numpy.ctypeslib.ndpointer(ctypes.c_int, flags = "C_CONTIGUOUS"))
            elif arg == long_arr:
                args += "long* %s" % name
                argtypes.append(numpy.ctypeslib.ndpointer(ctypes.c_long, flags = "C_CONTIGUOUS"))
            elif arg == double_arr:
                args += "double* %s" % name
                argtypes.append(numpy.ctypeslib.ndpointer(ctypes.c_double, flags = "C_CONTIGUOUS"))
            else:
                name_mangle = name
                while name_mangle in self.__args.keys():
                    name_mangle = "%s_" % name_mangle
                args += "void* %s" % name_mangle
                argtypes.append(ctypes.c_void_p)
                cast_code += "    shared_ptr<%s> %s = (*((shared_ptr<%s>*)%s));\n" % \
                  (self.__boost_classes[arg], name, self.__boost_classes[arg], name_mangle)

        code = \
    """
    %s

    // Keep SWIG happy
    namespace dolfin {
    }
    using namespace dolfin;

    extern "C" {
      int code(%s) {
    %s

    %s
        return 0;
      }
    }""" % (self.__includes, args, cast_code, self.__code)

        mod = instant.build_module(code = code,
          cppargs = dolfin.parameters["form_compiler"]["cpp_optimize_flags"],
          lddargs = "-ldolfin", include_dirs = self.__include_dirs,
          cmake_packages = ["DOLFIN"])
        path = os.path.dirname(mod.__file__)
        name = os.path.split(path)[-1]
        self.__lib = ctypes.cdll.LoadLibrary(os.path.join(path, "_%s.so" % name))
        self.__lib.code.argtypes = argtypes
        self.__lib.code.restype = int

        return

    def run(self, **kwargs):
        """
        Run the code. The keyword arguments form a list of name:variable pairs,
        with:
          name:     The name of a variable in the C++ code, which will be passed
                    from Python.
          variable: The variable to be passed to the C++ code. The type must be
                    consistent with the type passed to the constructor.
        """

        args = kwargs
        if not len(args) == len(self.__args) or not tuple(sorted(args.keys())) == tuple(sorted(self.__args.keys())):
            raise InvalidArgumentException("Invalid argument names")
        for name in args.keys():
            arg = args[name]
            if isinstance(arg, numpy.ndarray):
                if arg.dtype == "int32":
                    if not self.__args[name] == int_arr:
                        raise InvalidArgumentException("Argument %s is of invalid type" % name)
                elif arg.dtype == "int64":
                    if not self.__args[name] == long_arr:
                        raise InvalidArgumentException("Argument %s is of invalid type" % name)
                elif arg.dtype == "float64":
                    if not self.__args[name] == double_arr:
                        raise InvalidArgumentException("Argument %s is of invalid type" % name)
                else:
                    raise InvalidArgumentException("Argument %s is of invalid type" % name)
            elif not isinstance(arg, self.__args[name]):
                raise InvalidArgumentException("Argument %s is of invalid type" % name)

        if self.__lib is None:
            self.compile()

        largs = []
        for name in sorted(args.keys()):
            arg = args[name]
            if isinstance(arg, int):
                largs.append(ctypes.c_int(arg))
            elif isinstance(arg, float):
                largs.append(ctypes.c_double(arg))
            elif isinstance(arg, numpy.ndarray):
                assert(arg.dtype in ["int32", "int64", "float64"])
                largs.append(arg)
            else:
                assert(isinstance(arg, tuple(self.__boost_classes.keys())))
                largs.append(ctypes.c_void_p(int(arg.this)))

        ret = self.__lib.code(*largs)
        if not ret == 0:
            raise StateException("Non-zero return value: %i" % ret)

        return

class CellKernel(EmbeddedCpp):
    """
    A wrapper for short sections of embedded C++ code which iterate over cells.

    Constructor arguments:
      mesh:                 A Mesh.
      kernel_code:          Code executed for each cell in the mesh.
      initialisation_code:  Code executed before iterating over the cells.
      finalisation_code:    Code executed after iterating over the cells.
      includes:             Code which can, for example, be used to include header
                            files.
      include_dirs:         Header file directories.
    Remaining keyword arguments are as for the EmbeddedCpp constructor. An
    additional size_t variable cell is defined in the cell iteration, indicating
    the cell number.
    """

    def __init__(self, mesh, kernel_code, initialisation_code = "", finalisation_code = "", includes = "", include_dirs = [], **kwargs):
        if not isinstance(mesh, dolfin.Mesh):
            raise InvalidArgumentException("mesh must be a Mesh")
        if not isinstance(kernel_code, str):
            raise InvalidArgumentException("kernel_code must be a string")
        if not isinstance(initialisation_code, str):
            raise InvalidArgumentException("initialisation_code must be a string")
        if not isinstance(finalisation_code, str):
            raise InvalidArgumentException("finalisation_code must be a string")

        code = \
    """
    %s
        for(size_t cell = 0;cell < %i;cell++) {
    %s
        }
    %s""" % (initialisation_code, mesh.num_cells(), kernel_code, finalisation_code)

        EmbeddedCpp.__init__(self, code, includes = includes, include_dirs = include_dirs, **kwargs)

        return
