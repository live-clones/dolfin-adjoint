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


Binary packages (Ubuntu)
========================

Binary packages are currently available for Ubuntu users through the
`launchpad PPA`_.  To install dolfin-adjoint, do

.. code-block:: bash

   sudo apt-add-repository ppa:libadjoint/ppa
   sudo apt-get update
   sudo apt-get install python-dolfin-adjoint

which should install the latest stable version on your system.
Once that's done, why not try out the :doc:`tutorial <../documentation/tutorial>`?

.. _launchpad PPA: https://launchpad.net/~libadjoint/+archive/ppa


Docker images (all platforms and versions)
==========================================

`Docker <https://www.docker.com>`_ allows us to build and ship
consistent high-performance FEniCS installations with dolfin-adjoint for almost any
platform. To get started, follow these 2 steps:

#. Install Docker. Mac and Windows users should install the `Docker
   Toolbox <https://www.docker.com/products/docker-toolbox>`_ (this is
   a simple one-click install) and Linux users should `follow these
   instructions <https://docs.docker.com/linux/step_one/>`_.
#. Install the FEniCS Docker script::

    curl -s https://bitbucket.org/dolfin-adjoint/virtual/raw/master/bin/get.fenicsproject-adjoint | sh

If running on Mac or Windows, make sure you run this and other
commands inside the Docker Quickstart Terminal.

Once both Docker and the FEniCS Docker script have been installed, you can
easily start a FEniCS session with dolfin-adjoint by running the following
command::

    fenicsproject-adjoint run quay.io/dolfinadjoint/dolfin-adjoint

The FEniCS Docker script can also be used to create persistent sessions
(``fenicsproject-adjoint create myproject`` followed by ``fenicsproject-adjoint run
myproject``) or to run different versions of FEniCS (``fenicsproject-adjoint run
quay.io/dolfinadjoint/dev-dolfin-adjoint``). To see more options, run the
following command::

    fenicsproject-adjoint help

For more details and tips on how to work with FEniCS and Docker, see
our `FEniCS Docker page
<http://fenics-containers.readthedocs.org/en/latest/>`_.

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

Older versions
==============

An older version compatible with FEniCS 1.5 can be downloaded with:

.. code-block:: bash

   git clone -b dolfin-adjoint-1.5 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
   git clone -b libadjoint-1.5 https://bitbucket.org/dolfin-adjoint/libadjoint
