from dolfin import *
from dolfin_adjoint import *
import ufl

import numpy as np
from IPython import embed as key

# The difference function
def dif(l):
    return l('-') - l('+')

# Define mesh
Ne   = 2 # number of elements per cm
mesh = RectangleMesh(Point(0., 0.), Point(30.e-3, 20.e-3), 3*Ne, 2*Ne, "crossed")
n    = FacetNormal(mesh)

# Define functionspaces
U = VectorFunctionSpace(mesh, "DG", 3, dim = 2)
V = VectorFunctionSpace(mesh, "DG", 3, dim = 3)
Ds = FunctionSpace(mesh, "CG", 1)

# Define source and receiver
S = np.array(((10e-3, 0.), (20e-3, 0.), (10e-3, 20e-3), (20e-3, 20e-3), (0., 10e-3), (30e-3, 10e-3))) # Source coordinates
R = np.array(((6.25e-3, 12.5e-3),(16.25e-3, 12.5e-3),(23.75e-3, 12.5e-3),(6.25e-3, 7.5e-3),(13.75e-3, 7.5e-3),(23.75e-3, 7.5e-3))) # Receiver coordinates

# Forward solver
def forward(cl, ct, Forward=True, Record=False, Annotate=False):
    if Record: print "Recording reference ..........................."

    # Set material parameters
    rho = Constant(2700.) # [kg/m^3]
    C   = 6000      # Set max speed for numerical fluxes

    c11 = c22 = c33 = cl**2*rho #       [Pa]
    c44 = c55 = c66 = ct**2*rho #       [Pa]
    c12 = c13 = c23 = c11-2*c66 #       [Pa]

    # Set time stepping params
    dt = 1.e-8        # time step size
    DT = Constant(dt) # constant for UFL formulation
    t = dt            # initial time
    T = 1.e-5         # final time
    N = T/dt          # number of time steps

    # Test and trial functions
    q1 = TrialFunction(U)
    q2 = TrialFunction(V)
    l1 = TestFunction(U)
    l2 = TestFunction(V)

    # Excitation
    D     = "((0.5+a*pow((t-td), 2))*exp(a*pow((t-td), 2)))"
    apod  = "exp(-0.5*pow((x[0]-xs)/w, 2))*(x[1] == ys)"

    s = Constant(0.0)
    for i in range(S.shape[0]):
        s = Expression(("0.0", apod+"*"+D), w=3.e-3, td=2.e-6, \
                                            a=-(pi*6.e5)**2, xs =S[i,0], \
                                            ys = S[i,1], t = 0.0, degree = 3)
    zero2 = Expression(("0.0", "0.0"), degree = 1)
    zero3 = Expression(("0.0", "0.0", "0.0"), degree = 1)

    # Jacobian matrices
    Ax1 = as_matrix([[-1., 0. ], [0. , 0. ], [0. , -1.]])
    Ax2 = as_matrix([[-c11/rho, -c12/rho, 0], [0, 0, -c66/rho]])
    Ay1 = as_matrix([[0, 0], [0, -1], [-1., 0]])
    Ay2 = as_matrix([[0, 0, -c66/rho], [-c12/rho, -c22/rho, 0]])
    Gf = as_matrix([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    Gv = as_matrix([[1 , 0], [0 , 1]])

    # set inital values
    q10 = interpolate(zero2, U, name = "voldstate")
    q20 = interpolate(zero3, V, name = "foldstate")

    # Define fluxes on interior and exterior facets
    q1hat    = n[0]('-')*avg(Ax1*q10) + n[1]('-')*avg(Ay1*q10) + C*0.5*dif(q20)
    q1hatbnd = (n[0]*Ax1 + n[1]*Ay1)*(Gv*q10) + C*Gf*q20
    q2hat    = n[0]('-')*avg(Ax2*q20) + n[1]('-')*avg(Ay2*q20) + C*0.5*dif(q10)
    q2hatbnd = (n[0]*Ax2 + n[1]*Ay2)*(Gf*q20) + C*Gv*q10

    # Variational formulation
    F1 = inner(q1 - q10, l1)*dx \
       - DT*(inner(Ax2*q20, l1.dx(0)) + inner(Ay2*q20, l1.dx(1)))*dx \
       + DT*inner(q2hat, dif(l1))*dS \
       + DT*inner(q2hatbnd, l1)*ds \
       - DT*inner(s/rho, l1)*dx

    F2 = inner(q2 - q20, l2)*dx \
       - DT*(inner(Ax1*q10, l2.dx(0)) + inner(Ay1*q10, l2.dx(1)))*dx \
       + DT*inner(q1hat, dif(l2))*dS \
       + DT*inner(q1hatbnd, l2)*ds

    a1, L1 = lhs(F1), rhs(F1)
    a2, L2 = lhs(F2), rhs(F2)

    # assembling
    A1 = assemble(a1)
    A2 = assemble(a2)

    # LU solvers
    A1solve = LUSolver()
    A1solve.set_operator(A1)
    A2solve = LUSolver()
    A2solve.set_operator(A2)

    # Prepare function for solution
    vt = Function(U, name = "vstate", annotate = Annotate)
    ft = Function(V, name = "fstate", annotate = Annotate)

    # initiate time stepping
    tstep = 1
    solus = {}

    times = []
    if Annotate: adj_start_timestep()
    for i in range(R.shape[0]): solus[i] = []
    # Actual time stepping
    while t < T+.5*dt:
        s.t = t

        # first leap
        b1 = assemble(L1)
        A1solve.solve(vt.vector(), b1, annotate = Annotate)
        q10.assign(vt, annotate = Annotate)

        # second leap
        b2 = assemble(L2)
        A2solve.solve(ft.vector(), b2, annotate = Annotate)
        q20.assign(ft, annotate = Annotate)

        print "{:4.0f}/{:4.0f}: {:3.2e}".format(tstep, N, t)

        # make sure times match solus
        times.append(t)
        for i in range(R.shape[0]): solus[i].append(vt(R[i,:]))

        # increase time
        if Annotate: adj_inc_timestep(t, tstep > N)
        tstep += 1
        t = tstep*dt # more accurate than t += dt

    if Record:
        np.savetxt("receivers.txt", R)
        for i in range(R.shape[0]):
            np.savetxt("received_%04i.txt" %i, np.array(solus[i]))

    return q10, times, solus

opt_file = File("clt_opt.pvd")
cl_opt = Function(Ds)
def eval_cb(j, m):
    cl_opt.assign(m)
    opt_file << cl_opt
    print("objective = %15.10e " % j)

#------------------------------------------------------------------------------
def optimize():

    # Define the control
    cl = interpolate(Constant(6320.), Ds, name="cl")
    ct = Constant(3130.)

    # Execute first time to annotate and record the tape
    v, times, states = forward(cl, ct, Forward = True, Record = False, Annotate = True)

    #adj_html("forward.html", "forward")
    #adj_html("adjoint.html", "adjoint")

    # Load references
    recs = np.loadtxt("receivers.txt")
    Refs = []
    start = 1
    for i in range(recs.shape[0]):
      rec = np.loadtxt("received_%04i.txt" %i)
      refs = [Constant(x) for x in rec[start:len(times), -1]]
      Refs.append(refs)

    # Prepare the objective function
    start = 1
    reg = Constant(0e-6)*(inner(cl, cl)+inner(grad(cl), grad(cl)))*dx
    J = PointwiseFunctional(v, Refs, R, times[start:], u_ind=1, boost=1.e20, verbose=True, regform=reg)

    Jr = ReducedFunctional(J, Control(cl), eval_cb_post=eval_cb)
    problem = MinimizationProblem(Jr)


    parameters = { "type": "blmvm",
                   "max_it": 2000,
                   "fatol": 1e-100,
                   "frtol": 0.0,
                   "gatol": 5e-9,
                   "grtol": 0.0
                 }

    # Now construct the TAO solver and pass the Riesz map.
    solver = TAOSolver(problem, parameters=parameters, riesz_map=L2(Ds), prefix="opt")

    cl_opttt = solver.solve()
    File("output/cl_opttt.pvd") << cl_opttt

if __name__ == "__main__":
    # Record a reference solution
    if "-r" in sys.argv:
        Defs = np.array(((12.5e-3, 12.5e-3), (17.5e-3,7.5e-3)))
        defect = Constant(6320.)
        cl     = interpolate(defect, Ds, name="cl")
        for i in range(Defs.shape[0]):
            defect = Expression("1.+a*exp(-0.5*pow((x[0]-xs)/w, 2))*(x[1] == ys)", a =0.05, xs=Defs[i,0], ys=Defs[i,1], w=2e-3)
            cl *= interpolate(defect, Ds)

#        plot(cl, interactive = True)
        forward(cl, Constant(3130.), Forward = True, Record = True, Annotate = False)

    # Optimize controls
    if "-o" in sys.argv:
        optimize()
