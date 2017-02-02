"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from firedrake import *
from firedrake_adjoint import *

firedrake.parameters["adjoint"]["record_all"] = True
firedrake.parameters["adjoint"]["fussy_replay"] = False

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

class BurgersProblem(NonlinearVariationalProblem):
    def __init__(self, F, u, bc):
        NonlinearVariationalProblem.__init__(self, F, u)
        self.f = F
        self.jacob = derivative(F, u)
        self.bc = bc

    def F(self, b, x):
        assemble(self.f, tensor=b)
        self.bc.apply(b)

    def J(self, A, x):
        assemble(self.jacob, tensor=A)
        self.bc.apply(A)

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    bc = DirichletBC(V, 0.0, "on_boundary")
    burgers = BurgersProblem((Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx, u, bc)
    
    #solver = NewtonSolver()
    #solver.parameters["convergence_criterion"] = "incremental"
    #solver.parameters["relative_tolerance"] = 1e-6
    #tentative
    solver = NonlinearVariationalSolver(burgers, solver_parameters={'ksp_rtol': 1e-6})

    t = 0.0
    end = 0.2
    while (t <= end):
        solver.solve(burgers, u.vector(), annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    forward = main(ic, annotate=True)

    J = Functional(forward*forward*dx*dt[FINISH_TIME])
    m = Control(forward)
    dJdm = compute_gradient(J, m, forget = False)
    Jm = assemble(forward*forward*dx)

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm)
