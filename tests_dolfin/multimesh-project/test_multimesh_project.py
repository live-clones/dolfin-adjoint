from dolfin import *
from dolfin_adjoint import *

def test_project():
    mesh_0 = UnitSquareMesh(10, 10)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.build()

    # Define functionspace
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)

    g = MultiMeshFunction(V)
    g.vector()[:] = 1
    u0 = project(g, V, name="u0", annotate=True)

    J = Functional(u0**3*dX)
    m = Control(g)

    rf = ReducedFunctional(J, m)
    order = rf.taylor_test(g)
    assert order > 1.9

if __name__ == '__main__':
    test_project()


