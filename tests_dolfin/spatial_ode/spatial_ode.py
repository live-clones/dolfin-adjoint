from dolfin import *
from dolfin_adjoint import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

def main():
    """Test for solving a single ODE using PointIntegralSolver with
    spatially varying coefficents.
    """
    a = Constant(0.13)
    mesh = UnitIntervalMesh(2)
    Q = FunctionSpace(mesh, "CG", 1)
    aQ = Function(Q)
    aQ.vector()[:] = float(a)
    a = aQ

    time = Constant(0.0)
    I_s = Constant(0.0)

    VS = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    vs = Function(VS, name="vs")
    
    # What the cardiac ODE solver does
    (v, s) = split(vs)
    (w, q) = split(TestFunction(VS))
    rhs = ((v - s)*q - v*a*w)*dP()

    # Assign initial conditions
    scheme = GRL1(rhs, vs, time)

    # Initialize solver and update its parameters
    pi_solver = PointIntegralSolver(scheme)
    
    # Step once
    pi_solver.step(0.1)

    j = vs[0]*vs[0]*dx*dt[FINISH_TIME]
    J = Functional(j)
    
    dJdm = compute_gradient(J, Control(a))
    try:
        print float(dJdm)
    except:
        pass
        
if __name__ == "__main__":
    main()
