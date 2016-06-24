from dolfin import *
from dolfin_adjoint import *

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def solve_poisson():
    mesh = RectangleMesh(Point(-1.5,-0.5), Point(1.5,0.75), 40, 20)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    W = FunctionSpace(mesh, 'DG', 0)
    f = Constant(1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    A = assemble(a)
    b = assemble(L)
    noslip=Noslip()
    bc0 = DirichletBC(V, Constant(0), noslip)
    bc0.apply(A,b)
    u = Function(V)
    solve(A, u.vector(), b)

    plot(u)
    interactive()

if __name__=='__main__':
    solve_poisson()
