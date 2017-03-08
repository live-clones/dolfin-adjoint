__author__ = "Marie E. Rognes (meg@simula.no), 2017"

import math
import pylab
from cbcbeat import *

# For computing faster
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags
parameters["form_compiler"]["quadrature_degree"] = 4

def main():
    """Test for solving a single ODE using PointIntegralSolver with
    spatially varying coefficents.
    """

    a = Constant(0.13)
    v_peak = Constant(40.0)
    v_rest = -85.0
    v_amp = v_peak - v_rest

    mesh = UnitIntervalMesh(1)
    Q = FunctionSpace(mesh, "CG", 1)
    aQ = Function(Q)
    aQ.vector()[:] = float(a)
    a = aQ

    v_th = v_rest + a*v_amp

    time = Constant(0.0)
    I_s = Constant(0.0)

    num_states = 1
    VS = VectorFunctionSpace(mesh, "CG", 1, dim=2)

    vs = Function(VS, name="vs")
    
    # What the cardiac ODE solver does
    (v, s) = split(vs)
    (w, q) = split(TestFunction(VS))
    rhs = ((v - s)*q + v*v*(v - v_th)*w)*dP()

    # Assign initial conditions
    scheme = GRL1(rhs, vs, time)

    # Initialize solver and update its parameters
    pi_solver = PointIntegralSolver(scheme)
    
    # Solve and extract values
    k_n = 0.1
    T = 2*k_n

    while (float(time) < T):
        pi_solver.step(k_n)
        time.assign(float(time) + k_n)

    j = vs[0]*vs[0]*dx*dt[FINISH_TIME]
    J = Functional(j)
    
    dJdm = compute_gradient(J, Control(v_peak))
    try:
        print float(dJdm)
    except:
        pass
    plot(dJdm, mesh=mesh,interactive=True)
        
if __name__ == "__main__":

    main()
