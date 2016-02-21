import random

from firedrake import *
from firedrake_adjoint import *
import sys

mesh = UnitSquareMesh(4, 4)
V3 = FunctionSpace(mesh, "CG", 3)
V2 = FunctionSpace(mesh, "CG", 2)
firedrake.parameters["adjoint"]["record_all"] = True

def main(ic, annotate=False):
    soln = project(ic, V2, annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V3)
    soln = main(ic, annotate=True)

    adj_html("projection_forward.html", "forward")
    assert replay_dolfin(tol=1e-12, stop=True)

    J = Functional(soln*soln*dx*dt[FINISH_TIME])
    Jic = assemble(soln*soln*dx)
    dJdic = compute_gradient(J, FunctionControl(ic), forget=False)

    def J(ic):
        soln = main(ic, annotate=False)
        return assemble(soln*soln*dx)

    minconv = taylor_test(J, FunctionControl(ic), Jic, dJdic)
    assert minconv > 1.9
    info_green("Test passed")
