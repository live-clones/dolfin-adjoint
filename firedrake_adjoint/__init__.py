"""

The entire firedrake-adjoint interface should be imported with a single
call:

.. code-block:: python

  from firedrake import *
  from firedrake_adjoint import *

It is essential that the importing of the :py:mod:`firedrake_adjoint` module happen *after*
importing the :py:mod:`firedrake` module. firedrake-adjoint relies on *overloading* many of
the key functions of firedrake to achieve its degree of automation.
"""

__version__ = '2017.2.0'
__author__  = 'Simon Funke and Patrick Farrell'
__credits__ = ['Patrick Farrell', 'Simon Funke', 'David Ham', 'Marie Rognes']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 'simon@simula.no'

import sys
import firedrake
sys.modules['backend'] = firedrake
from dolfin_adjoint import *

firedrake.projection._solve = solve
