from dolfin import *
from dolfin_adjoint import *

class Dirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def test_move():
    mesh_0 = UnitSquareMesh(10, 10)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.build()

    # Define functionspace
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)
    u0 = MultiMeshFunction(V)
    u0.vector()[:] = 1
    u1 = MultiMeshFunction(V, name="u1")
    f = MultiMeshFunction(V)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0
    dt = 0.1

    # Define bilinear form
    a = u*v*dX + dt* ( inner(grad(u), grad(v))*dX \
        - dot(avg(grad(u)), jump(v, n))*dI \
        - dot(avg(grad(v)), jump(u, n))*dI \
        + alpha/h*jump(u)*jump(v)*dI \
        + beta*dot(jump(grad(u)), jump(grad(v)))*dO )

    # Define linear form
    L= u0*v*dX + dt*f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    # Solving linear system
    bound = Dirichlet()
    bc = MultiMeshDirichletBC(V, Constant(1), bound)
    bc.apply(A,b)
    solve(A, u1.vector(), b)

    J = Functional(u1**2*dX)
    m = [Control(u0), Control(f)]
    rf = ReducedFunctional(J, m)
    order = rf.taylor_test([u0, f])
    assert order > 1.9


if __name__ == '__main__':
    test_move()
