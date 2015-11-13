from dolfin import *
from dolfin_adjoint import *

parameters["adjoint"]["cache_factorizations"] = True

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m, n=0):
    u = interpolate(Constant(0.1), V, name="Solution")

    F = inner(u*u, test)*dx - inner(m+n, test)*dx
    solve(F == 0, u)
    F = inner(sin(u)*u*u*trial, test)*dx - inner(u**4, test)*dx
    solve(lhs(F) == rhs(F), u)

    return u

if __name__ == "__main__":
    m = interpolate(Constant(2.13), V, name="Parameter1")
    n = interpolate(Constant(0.0), V, name="Parameter2")
    u = main(m,n)

    parameters["adjoint"]["stop_annotating"] = True

    J = Functional((inner(u, u))**3*dx + inner(m+n, m+n)*dx, name="NormSquared")
    Jm = assemble(inner(u, u)**3*dx + inner(m+n, m+n)*dx)

    controls = [Control(m)]

    dJdm = compute_gradient(J, controls, forget=None)
    HJm  = hessian(J, controls, warn=False)

    def Jhat(m, n=0):
        u = main(m, n)
        return assemble(inner(u, u)**3*dx + inner(m+n, m+n)*dx)

    dJdm0 = dJdm[0]
    HJm0 = lambda d: HJm([d])[0]

    # Argh: Taylor test does not work for multi controls yet ...
    direction = interpolate(Constant(0.1), V)
    minconv = taylor_test(Jhat, controls[0], Jm, dJdm0, HJm=HJm0,
                          perturbation_direction=direction)
    assert minconv > 2.9
