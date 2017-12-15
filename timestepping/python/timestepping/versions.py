#!/usr/bin/env python3

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

__all__ = \
  [
    "system_info"
  ]

def system_info():
    """
    Print system information and assorted library versions.
    """

    import platform
    import socket
    import time

    import dolfin
    import FIAT
    import ffc
    import instant
    import numpy
    import scipy
    import ufl

    dolfin.info("Date / time    : %s" % time.ctime())
    dolfin.info("Machine        : %s" % socket.gethostname())
    dolfin.info("Platform       : %s" % platform.platform())
    dolfin.info("Processor      : %s" % platform.processor())
    dolfin.info("Python version : %s" % platform.python_version())
    dolfin.info("NumPy version  : %s" % numpy.__version__)
    dolfin.info("SciPy version  : %s" % scipy.__version__)
    try:
        import mpi4py
        dolfin.info("MPI4Py version : %s" % mpi4py.__version__)
    except ImportError:
        pass
    try:
        import petsc4py.PETSc
        info = petsc4py.PETSc.Sys.getVersionInfo()
        dolfin.info("PETSc version  : %i.%i.%ip%i%s" % (info["major"], info["minor"], info["subminor"], info["patch"], "" if info["release"] else "dev"))
    except ImportError:
        pass
    try:
        import vtk
        dolfin.info("VTK version    : %s" % vtk.vtkVersion().GetVTKVersion())
    except ImportError:
        pass
    dolfin.info("DOLFIN version : %s" % dolfin.__version__)
    dolfin.info("FIAT version   : %s" % FIAT.__version__)
    dolfin.info("FFC version    : %s" % ffc.__version__)
    dolfin.info("Instant version: %s" % instant.__version__)
    dolfin.info("UFL version    : %s" % ufl.__version__)

    return
