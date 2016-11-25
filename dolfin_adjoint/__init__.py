"""

The entire dolfin-adjoint interface should be imported with a single
call:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint import *

It is essential that the importing of the :py:mod:`dolfin_adjoint` module happen *after*
importing the :py:mod:`dolfin` module. dolfin-adjoint relies on *overloading* many of
the key functions of dolfin to achieve its degree of automation.
"""

__version__ = '2016.1.0'
__author__  = 'Patrick Farrell and Simon Funke'
__credits__ = ['Patrick Farrell', 'Simon Funke', 'David Ham', 'Marie Rognes']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 'simon@simula.no'

import sys
if not 'backend' in sys.modules:
    import dolfin
    sys.modules['backend'] = dolfin
backend = sys.modules['backend']

from . import options
from . import solving
from . import assembly
from . import expressions
from . import utils
from . import assignment
from . import functional
from . import split_annotation
from . import constant

if backend.__name__ == "dolfin":
    from . import lusolver
    backend.comm_world = dolfin.mpi_comm_world()
else:
    from mpi4py import MPI
    backend.comm_world = MPI.COMM_WORLD

from . import gst
from . import function

from . import optimization
from . import reduced_functional
from .optimization import optimization
if backend.__name__ == "dolfin":
    from . import genericmatrix
    from . import genericvector
    from . import optimization
    from . import reduced_functional
    from . import pointwise_functional
    from .optimization import optimization

from .ui import *
