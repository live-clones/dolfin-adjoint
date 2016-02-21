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

__version__ = '1.6'
__author__  = 'Patrick Farrell and Simon Funke'
__credits__ = ['Patrick Farrell', 'Simon Funke', 'David Ham', 'Marie Rognes']
__license__ = 'LGPL-3'
__maintainer__ = 'Patrick Farrell'
__email__ = 'patrick.farrell@maths.ox.ac.uk'

import sys
if not 'backend' in sys.modules:
    import dolfin
    sys.modules['backend'] = dolfin
backend = sys.modules['backend']

import options
import solving
import assembly
import expressions
import utils
import assignment
import functional
import split_annotation

if backend.__name__ == "dolfin":
    import lusolver

import gst
import function

if backend.__name__ == "dolfin":
    import genericmatrix
    import genericvector
    import optimization
    import reduced_functional
    import pointwise_functional
    from optimization import optimization


from ui import *
