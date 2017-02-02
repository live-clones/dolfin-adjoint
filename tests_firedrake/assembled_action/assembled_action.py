from firedrake import *
from firedrake_adjoint import *

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)

def main(data):
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = inner(u, v)*dx
    M = assemble(mass)

    #rhs = M*data.vector()
    rhs = Function(V)
    with data.dat.vec_ro as x, rhs.dat.vec as y:
        M.petscmat.mult(x, y)
    
    soln = Function(V)

    solve(M, soln.vector(), rhs)
    return soln

if __name__ == "__main__":
    data = Function(V, name="Data")
    data.vector()[0] = 1.0

    soln = main(data)
    parameters["adjoint"]["stop_annotating"] = True

    J = Functional(inner(soln, soln)*dx*dt[FINISH_TIME])
    j = assemble(inner(soln, soln)*dx)
    dJdic = compute_gradient(J, FunctionControl("Data"), forget=False)

    def Jhat(data):
        soln = main(data)
        return assemble(soln*soln*dx)

    minconv = taylor_test(Jhat, FunctionControl("Data"), j, dJdic)
    assert minconv > 1.9
