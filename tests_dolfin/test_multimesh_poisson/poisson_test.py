from dolfin import *

# Boundary of propeller after overwrite from the other boundaries
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
    
# class InflowBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(x[0],-1.5)

# class OutflowBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(x[0],1.5)

# class NonslipBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(x[1],-0.75) or near(x[1],0.75)

    
def solve_poisson():
    mesh_0 = RectangleMesh(Point(-1.5, -0.75), Point(1.5, 0.75), 40, 20)
    propeller = Mesh("./propeller_2d_coarse.xml.gz")
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(propeller)
    multimesh.build()
    
    # Create function space
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form
    a = dot(grad(u), grad(v))*dX \
      - dot(avg(grad(u)), jump(v, n))*dI \
      - dot(avg(grad(v)), jump(u, n))*dI \
      + alpha/h*jump(u)*jump(v)*dI \
      + beta*dot(jump(grad(u)), jump(grad(v)))*dO

    # Define linear form
    L = f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    
    # Creating subdomains
    # subdomains = FacetFunction('size_t',mesh_0)
    # inb = InflowBoundary()
    # outb = OutflowBoundary()
    # noslip = NonslipBoundary()
    # prop = Noslip()
    # inb.mark(subdomains,10)
    # plot(mesh_0)
    # bc0 = MultiMeshDirichletBC(V, Constant(0), prop,
    #                            exclude_overlapped_boundaries=True)
    # bc1 = MultiMeshDirichletBC(V, Constant(0),inb)
    # bc2 = MultiMeshDirichletBC(V, Constant(0), outb)
    # bc3 = MultiMeshDirichletBC(V, Constant(0), noslip)
    # bc0.apply(A, b)
    # bc1.apply(A, b)
    # bc2.apply(A, b)
    # bc3.apply(A, b)
    # u = MultiMeshFunction(V)
    # solve(A, u.vector(), b)
    # plot(u.part(0), title='u0')
    # interactive()
    # plot(u.part(1), title='u1')
    # interactive()
    # plot(u.part(2), title='u2')
    # plot(multimesh)
    # interactive()
    for f in n:
        print f
    
if __name__ == '__main__':
    solve_poisson()
