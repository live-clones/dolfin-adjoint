"""This demo program solves Poisson's equation

  - div grad u(x, y) = 0

on the unit square with boundary conditions given by:

  u(0, y) = 0
  v(1, y) = 42

Homogeneous Neumann boundary conditions are applied naturally on the
other two sides of the domain.

This has the analytical solution

  u(x, y) = 42*x[1]
"""
from __future__ import print_function
from firedrake import *
from firedrake_adjoint import *
import pytest


@pytest.fixture
def V():
    mesh = UnitSquareMesh(10, 10)
    return FunctionSpace(mesh, "CG", 1)


def model(s, V):
    # Define variational problem
    u = Function(V, name="u")
    v = TestFunction(V)
    a = dot(grad(v), grad(u)) * dx + s * v * dx

    bcs = [DirichletBC(V, 0, 1),
           DirichletBC(V, 42, 2)]

    # Compute solution
    solve(a == 0, u, bcs=bcs)

    f = Function(V, name="f")
    f.interpolate(Expression("42*x[1]"))

    return assemble(dot(u - f, u - f) * dx), u, f


def test_poisson_bc(V):
    s = Function(V, name="s")
    s.assign(1)

    print("Running forward model")
    j, u, f = model(s, V)

    adj_html("forward.html", "forward")
    print("Replaying forward model")
    assert replay_dolfin(tol=1.e-14, stop=True)

    J = Functional(inner(u - f, u - f) * dx * dt[FINISH_TIME])
    m = FunctionControl(s)

    print("Running adjoint model")
    dJdm = compute_gradient(J, m, forget=None)

    parameters["adjoint"]["stop_annotating"] = True

    Jhat = lambda s: model(s, V)[0]
    conv_rate = taylor_test(Jhat, m, j, dJdm, seed=1e-3)
    assert conv_rate > 1.85
