""" Solves a MMS problem with smooth control """
from firedrake import *
from firedrake_adjoint import *
import pytest

try:
    from petsc4py import PETSc
except ImportError:
    pass


def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)


@pytest.mark.skipif("petsc4py.PETSc" not in sys.modules or not hasattr(PETSc, "TAO"),
                    reason="PETSc bindings with TAO support unavailable")
def test_optimization_tao():
    n = 100
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')

    x = SpatialCoordinate(mesh)

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    J = Functional((inner(u-u_d, u-u_d))*dx*dt[FINISH_TIME])

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation
    rf = ReducedFunctional(J, FunctionControl(m, value=m))
    problem = MinimizationProblem(rf)
    opts = PETSc.Options()
    opts["tao_monitor"] = None
    opts["tao_view"] = None
    opts["tao_nls_ksp_type"] = "gltr"
    opts["tao_nls_pc_type"] = "none"
    opts["tao_ntr_pc_type"] = "none"
    parameters = {'method': 'nls',
                  'max_it': 20,
                  'fatol': 0.0,
                  'frtol': 0.0,
                  'gatol': 1e-9,
                  'grtol': 0.0
                  }

    solver = TAOSolver(problem, parameters=parameters)
    m_opt = solver.solve()

    solve_pde(u, V, m_opt)

    x, y = SpatialCoordinate(mesh)
    # Define the analytical expressions
    m_analytic = sin(pi*x)*sin(pi*y)
    u_analytic = 1.0/(2*pi*pi)*sin(pi*x)*sin(pi*y)

    # Compute the error
    control_error = sqrt(assemble((m_analytic - m_opt)**2*dx))
    state_error = sqrt(assemble((u_analytic - u)**2*dx))

    assert control_error < 0.01
    assert state_error < 1e-5
    # Check that values are below the threshold
    tao_p = solver.get_tao()
    assert tao_p.gnorm < 1e-9
    assert tao_p.getIterationNumber() <= 20
