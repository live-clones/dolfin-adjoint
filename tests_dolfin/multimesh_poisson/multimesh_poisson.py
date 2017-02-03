from dolfin import *
from dolfin_adjoint import *

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def solve_poisson_moola(N,element,degree):
    # Creating mesh
    import moola
    mesh_0 = RectangleMesh(Point(0,0), Point(1,1), N,N)
    mesh_1 = RectangleMesh(Point(0.25, 0.25),  Point(0.75, 0.75), N,N)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()

    # Create function space for the temperature
    V = MultiMeshFunctionSpace(multimesh, "CG", 1)
    W = MultiMeshFunctionSpace(multimesh, element, degree)

    # Initialize boundary condition on state
    boundary = Boundary()
    bc0 = MultiMeshDirichletBC(V, Constant(0), boundary)

    # Initial guess for source distribution
    bcf = MultiMeshDirichletBC(W, Constant(0), boundary)
    fex = Expression('x[0]+x[1]',degree=1)
    f = project(fex, W, name="Control")
    bc0.apply(f.vector())

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form and linear form
    a = dot(grad(u), grad(v))*dX \
        - dot(avg(grad(u)), jump(v, n))*dI \
        - dot(avg(grad(v)), jump(u, n))*dI \
        + alpha/h*jump(u)*jump(v)*dI  \
        + beta*dot(jump(grad(u)), jump(grad(v)))*dO
    L= f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)
    bc0.apply(A,b)
    
    # Solving linear system
    u = MultiMeshFunction(V, name="State")
    solve(A, u.vector(), b, 'lu')

    # Defining functional and desired temperature profile
    alpha1 = Constant(1e-6)
    x = SpatialCoordinate(multimesh)
    d = 2*pi**2*alpha1*sin(pi*x[0])*sin(pi*x[1]) + sin(pi*x[0])*sin(pi*x[1])/(2*pi**2)
    J = Functional(0.5*inner(u-d,u-d)*dX + alpha1/2.*f**2*dX)
    control = Control(f)
    rf = ReducedFunctional(J,control)

    # Moola optimization
    problem = MoolaOptimizationProblem(rf)
    f_moola = moola.DolfinPrimalVector(f, inner_product="L2")
    solver = moola.BFGS(problem, f_moola, options={'jtol': 1e-30,
                                                   'rjtol':1e-30,
                                                   'rgtol':1e-30,
                                                   'gtol':5e-11,
                                                   'Hinit': "default",
                                                   'maxiter': 200,
                                                   'mem_lim': 10})

    sol = solver.solve()
    f_opt = sol['control'].data

    # Computing u with optimal control
    L = f_opt*v*dX
    b = assemble_multimesh(L)
    bc0.apply(b)
    solve(A, u.vector(), b, 'lu')

    # Analytical solutions to problem
    Ex = MultiMeshFunctionSpace(multimesh, "CG", 3)
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=3)
    u_analytic = project(u_analytic, Ex)
    f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    f_analytic = project(f_analytic, Ex)

    # L2 errors
    control_error = sqrt(assemble_multimesh((f_opt - f_analytic)**2*dX))
    state_error = sqrt(assemble_multimesh((u - u_analytic)**2*dX))
    return 1./N, control_error

def solve_poisson_scipy(N,element,degree):
    # Creating mesh
    mesh_0 = RectangleMesh(Point(0,0), Point(1,1), N,N)
    mesh_1 = RectangleMesh(Point(0.25, 0.25),  Point(0.75, 0.75), N,N)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()

    # Create function space for the temperature
    V = MultiMeshFunctionSpace(multimesh, "CG", 1)
    W = MultiMeshFunctionSpace(multimesh, element, degree)

    # Initialize boundary condition on state
    boundary = Boundary()
    bc0 = MultiMeshDirichletBC(V, Constant(0), boundary)

    # Initial guess for source distribution
    bcf = MultiMeshDirichletBC(W, Constant(0), boundary)
    fex = Expression('x[0]+x[1]',degree=1)
    f = project(fex, W, name="Control")
    bc0.apply(f.vector())

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form and linear form
    a = dot(grad(u), grad(v))*dX \
        - dot(avg(grad(u)), jump(v, n))*dI \
        - dot(avg(grad(v)), jump(u, n))*dI \
        + alpha/h*jump(u)*jump(v)*dI  \
        + beta*dot(jump(grad(u)), jump(grad(v)))*dO
    L= f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)
    bc0.apply(A,b)

    # Solving linear system
    u = MultiMeshFunction(V, name="State")
    solve(A, u.vector(), b, 'lu')

    # Defining functional and desired temperature profile
    alpha1 = Constant(1e-6)
    x = SpatialCoordinate(multimesh)
    d = 2*pi**2*alpha1*sin(pi*x[0])*sin(pi*x[1]) + sin(pi*x[0])*sin(pi*x[1])/(2*pi**2)
    J = Functional(0.5*inner(u-d,u-d)*dX + alpha1/2.*f**2*dX)
    control = Control(f)
    rf = ReducedFunctional(J,control)

    # Scipy optimization
    f_opt = minimize(rf, tol=1e-16, options={"ftol":1e-15})

    # Computing u with optimal control
    L = f_opt*v*dX
    b = assemble_multimesh(L)
    bc0.apply(b)
    solve(A, u.vector(), b, 'lu')

    # Analytical solutions to problem
    Ex = MultiMeshFunctionSpace(multimesh, "CG", 3)
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=3)
    u_analytic = project(u_analytic, Ex)
    f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    f_analytic = project(f_analytic, Ex)

    # L2 errors
    control_error = sqrt(assemble_multimesh((f_opt - f_analytic)**2*dX))
    state_error = sqrt(assemble_multimesh((u - u_analytic)**2*dX))
    return 1./N, control_error
