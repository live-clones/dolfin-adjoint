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

import glob
import os

import dolfin
import ffc
import instant
import numpy
import scipy
import ufl
import vtk

import fenics_versions

from embedded_cpp import *
from fenics_versions import *

__all__ = \
  fenics_versions.__all__ + \
  [
    "petsc_version",
    "system_info"
  ]


def petsc_version():
    """
    Attempt to determine the current PETSc version, and return a Version if this
    is successful. Otherwise, return None.
    """

    if "PETSC_DIR" in os.environ:
        petsc_dir = os.environ["PETSC_DIR"]
    else:
        paths = sorted(glob.glob("/usr/lib/petscdir/*"), reverse = True)
        petsc_dir = "/usr"
        for path in paths:
            if os.path.isdir(path):
                petsc_dir = path
                break
    if "PETSC_ARCH" in os.environ:
        petsc_arch = os.environ["PETSC_ARCH"]
    else:
        petsc_arch = ""

    version = None
    for include_dir in [os.path.join(petsc_dir, petsc_arch, os.path.pardir, "include"),
                        os.path.join(petsc_dir, petsc_arch, "include"),
                        "/usr/include/petsc"]:
        if os.path.isfile(os.path.join(include_dir, "petscversion.h")):
            version = numpy.empty(4, dtype = numpy.int)
            EmbeddedCpp(includes = \
              """
              #include "petscversion.h"
              """,
              code = \
              """
              version[0] = PETSC_VERSION_MAJOR;
              version[1] = PETSC_VERSION_MINOR;
              version[2] = PETSC_VERSION_SUBMINOR;
              version[3] = PETSC_VERSION_PATCH;
              """, include_dirs = [include_dir], version = long_arr).run(version = version)
            break

    if version is None:
        return None
    else:
        return Version(version)

def system_info():
    """
    Print system information and assorted library versions.
    """

    import platform
    import socket
    import time

    import FIAT
    import instant
    import ufc

    dolfin.info("Date / time    : %s" % time.ctime())
    dolfin.info("Machine        : %s" % socket.gethostname())
    dolfin.info("Platform       : %s" % platform.platform())
    dolfin.info("Processor      : %s" % platform.processor())
    dolfin.info("Python version : %s" % platform.python_version())
    dolfin.info("NumPy version  : %s" % numpy.__version__)
    dolfin.info("SciPy version  : %s" % scipy.__version__)
    dolfin.info("VTK version    : %s" % vtk.vtkVersion().GetVTKVersion())
    dolfin.info("DOLFIN version : %s" % dolfin.__version__)
    dolfin.info("FIAT version   : %s" % FIAT.__version__)
    try:
        import ferari
        dolfin.info("FErari version : %s" % ferari.VERSION)
    except ImportError:
        pass
    dolfin.info("FFC version    : %s" % ffc.__version__)
    dolfin.info("Instant version: %s" % instant.__version__)
    try:
        import SyFi
        dolfin.info("SyFi version   : %i.%i" % (SyFi.version_major, SyFi.version_minor))
    except ImportError:
        pass
    dolfin.info("UFC version    : %s" % ufc.__version__)
    dolfin.info("UFL version    : %s" % ufl.__version__)
    try:
        import viper
        dolfin.info("Viper version  : %s" % viper.__version__)
    except ImportError:
        pass
    petsc_ver = petsc_version()
    if petsc_ver is None:
        dolfin.info("PETSc version  : Unknown")
    else:
        dolfin.info("PETSc version  : %i.%i.%ip%i" % petsc_ver.tuple())

    return
