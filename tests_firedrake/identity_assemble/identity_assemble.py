from firedrake import *
from firedrake_adjoint import *
# Create mesh and define function space
n = 5
mesh = UnitSquareMesh(2 ** n, 2 ** n)
V = FunctionSpace(mesh, "CG", 1)

def model(s):
    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(v, u) * dx
    L = s * v * dx

    # Compute solution
    x = Function(V, name="State")
    solve(a == L, x)
    j = assemble(x**2 * dx)
    return j, x

if __name__ == '__main__':

    s = Function(V)
    s.interpolate(Expression("1"))

    print "Running forward model"
    j, x = model(s)

    print "Replaying forward model"
    assert replay_dolfin(tol=1e-15, stop=True)

    J = Functional(x**2*dx*dt[FINISH_TIME])
    m = FunctionControl(s)

    print "Running the adjoint model"
    for i in compute_adjoint(J, forget=None):
        pass

    print "Computing the gradient with the adjoint model"
    dJdm = compute_gradient(J, m, forget=None)

    parameters["adjoint"]["stop_annotating"] = True

    Jhat = lambda s: model(s)[0]
    conv_rate = taylor_test(Jhat, m, j, dJdm)
    assert conv_rate > 1.9
    info_green("Test passed")
