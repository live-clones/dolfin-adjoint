.. _download:

**************************
Downloading dolfin-adjoint
**************************

Dependencies
============

Mandatory dependencies:
-----------------------

- `FEniCS`_, version 1.6. For installation instructions for FEniCS, see `their installation instructions`_.

- `libadjoint`_, version 1.6. This is a library written in C that manipulates the tape of the forward model to derive the associated adjoint equations.

Optional dependencies:
----------------------

- `SLEPc`_. This is necessary if you want to conduct :doc:`generalised stability analyses <../documentation/gst>`.

- `IPOPT`_ and `pyipopt`_: This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Install with `pip install moola`. Note: still experimental.

.. _FEniCS: http://fenicsproject.org
.. _libadjoint: http://bitbucket.org/dolfin-adjoint/libadjoint
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _pyipopt: https://github.com/xuy/pyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download

Virtual machine
===============

If you'd like to try dolfin-adjoint out without any installation headaches,
try out `the FENICS VirtualBox virtual machine with FEniCS and dolfin-adjoint pre-installed
<http://fenicsproject.org/pub/virtual/fenics-latest.ova>`_. Here are
the instructions:

* Download and install VirtualBox from https://www.virtualbox.org, or from your operating system.
* Download the `virtual machine <http://fenicsproject.org/pub/virtual/fenics-latest.ova>`_.
* Start VirtualBox, click on "File -> Import Appliance", select the virtual machine image and click on "Import".
* Select the "dolfin-adjoint VM" and click on "Start" to boot the machine.
* For installing new software you need the login credentials:

  * Username: fenics
  * Password: fenics

Binary packages
===============

Binary packages are currently available for Ubuntu users through the
`launchpad PPA`_.  To install dolfin-adjoint, do

.. code-block:: bash

   sudo apt-add-repository ppa:libadjoint/ppa
   sudo apt-get update
   sudo apt-get install python-dolfin-adjoint

which should install the latest stable version on your system.
Once that's done, why not try out the :doc:`tutorial <../documentation/tutorial>`?

.. _launchpad PPA: https://launchpad.net/~libadjoint/+archive/ppa

Docker container
============

A Docker container with dolfin-adjoint compiled against the latest FEniCS, PETSc and SLEPc 
is available on `Dockerhub<https://hub.docker.com/r/fenicsproject/dolfin-adjoint>`. 
Here are the instructions:

* Download and install Docker for your platform following the instructions at https://docker.com
* To download and run a dolfin-adjoint container:

.. code-block:: bash

   docker run -ti fenicsproject/dolfin-adjoint
   
You will be presented with a bash prompt where you can run dolfin-adjoint.

From source
===========

The latest stable release of dolfin-adjoint and libadjoint is **version 1.6** which is compatible with FEniCS 1.6. Download links:

* libadjoint:

.. code-block:: bash

   git clone -b libadjoint-1.6 https://bitbucket.org/dolfin-adjoint/libadjoint

* dolfin-adjoint:

.. code-block:: bash

   git clone -b dolfin-adjoint-1.6 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint

The **development version** is available with the following
command:

.. code-block:: bash

   git clone https://bitbucket.org/dolfin-adjoint/libadjoint
   git clone https://bitbucket.org/dolfin-adjoint/dolfin-adjoint

As dolfin-adjoint is a pure Python module, once its dependencies are
installed the development version can be used without system-wide
installation via

.. code-block:: bash

   export PYTHONPATH=<path to dolfin-adjoint>:$PYTHONPATH

libadjoint needs to be compiled with:

.. code-block:: bash

   cd libadjoint
   mkdir build; cd build
   cmake -DCMAKE_INSTALL_PREFIX=<install directory> ..
   make install


Contributions (such as handling new features of FEniCS, or new test
cases or examples) are very welcome.

Older versions
==============

An older version compatible with FEniCS 1.5 can be downloaded with:

.. code-block:: bash

   git clone -b dolfin-adjoint-1.5 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
   git clone -b libadjoint-1.5 https://bitbucket.org/dolfin-adjoint/libadjoint
