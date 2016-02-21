.. _dolfin-adjoint-tutorial:

.. py:currentmodule:: dolfin_adjoint

===========
First steps
===========

********
Foreword
********

If you have never used the FEniCS system before, you should first read
`their tutorial`_.  If you're not familiar with adjoints and their
uses, see the :doc:`background <maths/index>`.

***************
A first example
***************

Let's suppose you are interested in solving the nonlinear
time-dependent Burgers equation:

.. math::

    \frac{\partial \vec u}{\partial t} - \nu \nabla^2 \vec u + \vec u \cdot \nabla \vec u = 0,

subject to some initial and boundary conditions.


A forward model that solves this problem with P2 finite elements might
look as follows:

.. literalinclude:: ../_static/tutorial1.py

|more| You can `download the source code`_ and follow along as we
adjoin this code.

The first change necessary to adjoin this code is to import the
dolin-adjoint module **after** loading dolfin:

.. code-block:: python

    from dolfin import *
    from dolfin_adjoint import *

The reason why it is necessary to do it afterwards is because
dolfin-adjoint overloads many of the dolfin API functions to
understand what the forward code is doing.  In this particular case,
the :py:func:`solve <dolfin_adjoint.solve>` function and
:py:meth:`assign <dolfin_adjoint.Function.assign>` method have been
overloaded:

.. code-block:: python
   :emphasize-lines: 2,3

    while (t <= end):
        solve(F == 0, u_next, bc)
        u.assign(u_next)

The dolfin-adjoint versions of these functions will *record* each step
of the model, building an *annotation*, so that it can *symbolically
manipulate* the recorded equations to derive the tangent linear and
adjoint models.  Note that no user code had to be changed: it happens
fully automatically.

In order to talk about adjoints, one needs to consider a particular
functional. While dolfin-adjoint supports arbitrary functionals, let
us consider a simple nonlinear example.  Suppose our functional of
interest is the square of the norm of the final velocity:

.. math::

    J(u) = \int_{\Omega} \left\langle u(T), u(T) \right\rangle \ \textrm{d}\Omega,

or in code:

.. code-block:: python

    J = Functional(inner(u, u)*dx*dt[FINISH_TIME]).

Here, multiplying by :py:data:`*dt[FINISH_TIME]` indicates that the
functional is to be evaluated at the final time.

|more| If the functional were to be an integral over time, one could
multiply by :py:data:`*dt`. This requires some more annotation; see
the documentation for :py:func:`adj_inc_timestep`. For how to express
more complex functionals, see the documentation on :doc:`expressing
functionals <functionals>`.

The dolfin-adjoint software has several drivers, depending on
precisely what the user requires.  The highest-level interface is to
compute the gradient of the functional with respect to some
:py:class:`Control`. For example, suppose we wish to compute the
gradient of :math:`J` with respect to the initial condition for
:math:`u`, using the adjoint.  We can do this with the following code:

.. code-block:: python

    dJdic = compute_gradient(J, Control(u))

where :py:class:`Control
<dolfin_adjoint.Control>` indicates that the
gradient should be computed with respect to the initial condition of that function. This
single function call differentiates the model, assembles each adjoint
equation in turn, and then uses the adjoint solutions to compute the
requested gradient.

Other :py:class:`Controls` are possible. For example, to compute the
gradient of the functional :math:`J` with respect to the diffusivity
:math:`\nu`:

.. code-block:: python

    dJdnu = compute_gradient(J, Control(nu))

Note that by default,
:py:func:`compute_gradient <dolfin_adjoint.compute_gradient>`
deallocates all of the forward solutions it can as it goes along, to
minimise the memory footprint: however, if you try to run the adjoint
twice, it will give an error because it no longer has the necessary
forward variables:

.. code-block:: python

    >>> dJdic = compute_gradient(J, Control(u))
    >>> dJdnu = compute_gradient(J, Control(nu))
    Traceback (most recent call last):
    ...
    libadjoint.exceptions.LibadjointErrorNeedValue: Need a value for
       w_2:1:6:Forward, but don't have one recorded.

w_2 refers to the Function :py:data:`u`, and 1:6 means the sixth
(last) iteration associated with timestep 1; i.e., libadjoint is
telling us that it needs the terminal velocity, but it doesn't have
it, as it's been deallocated already in the first call to
:py:func:`compute_gradient <dolfin_adjoint.compute_gradient>`.  To
tell :py:func:`compute_gradient <dolfin_adjoint.compute_gradient>` not
to deallocate the forward solutions as it goes along, pass
:py:data:`forget=False`:

.. literalinclude:: ../_static/tutorial2.py

Observe how the changes required from the original forward code to the
adjoined version are very small: with only four lines added to the
original code, we are able to compute the gradient information.

|more| If you have been following along, you can `download the
adjoined Burgers' equation code`_ and compare your results.

Other interfaces are available to manually compute the adjoint and
tangent linear solutions. For details, see the section on
:doc:`lower-level interfaces <misc>`.

Once you have computed the gradient, how do you know if it is correct?
If you were to pass an incorrect gradient to an optimisation
algorithm, the convergence would be hampered or it may fail
entirely. Therefore, before using any gradients, you should satisfy
yourself that they are correct. dolfin-adjoint offers easy routines to
rigorously verify the computed results, which is the topic of the
:doc:`next section <verification>`.

.. _their tutorial: http://fenicsproject.org/documentation
.. _download the source code: ../_static/tutorial1.py
.. _download the adjoined Burgers' equation code: ../_static/tutorial2.py

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
