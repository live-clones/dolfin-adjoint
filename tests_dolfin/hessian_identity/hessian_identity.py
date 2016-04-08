from dolfin import *
from dolfin_adjoint import *

parameters["adjoint"]["cache_factorizations"] = True

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m):
    u = interpolate(Constant(0.1), V, name="Solution")

    F = inner(u*u, test)*dx - inner(m, test)*dx
    solve(F == 0, u)
    F = inner(sin(u)*u*u*trial, test)*dx - inner(u**4, test)*dx
    solve(lhs(F) == rhs(F), u)

    return u

if __name__ == "__main__":
    m = interpolate(Constant(2.13), V, name="Parameter")
    u = main(m)

    parameters["adjoint"]["stop_annotating"] = True

    J = Functional((inner(u, u))**3*dx + inner(m, m)*dx, name="NormSquared")
    Jm = assemble(inner(u, u)**3*dx + inner(m, m)*dx)
    dJdm = compute_gradient(J, Control(m), forget=None)
    HJm  = hessian(J, Control(m), warn=False)

    def Jhat(m):
        u = main(m)
        return assemble(inner(u, u)**3*dx + inner(m, m)*dx)

    minconv = taylor_test(Jhat, Control(m), Jm, dJdm, HJm=HJm,
                          perturbation_direction=interpolate(Constant(0.1), V))
    assert minconv > 2.9
