from firedrake import *
from firedrake_adjoint import *
dt_meas = dt

fexp = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)*sin(t)", t=0, degree=4)
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)

dt = Constant(0.1)
T = 1.0

def main():
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)

    # Create a list with all control functions
    ctrls = {}
    t = float(dt)
    while t <= T:
        fexp.t = t
        ctrls[t] = project(fexp, V, name="f_{}".format(t), annotate=True)
        t += float(dt)

    u_0 = Function(V, name="Solution")
    u_1 = Function(V, name="NextSolution")

    F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx
    a, L = lhs(F), rhs(F)
    #import ipdb
    #ipdb.set_trace()
    bc = DirichletBC(V, 1.0, "on_boundary")

    t = float(dt)
    adj_start_timestep(time=t)
    while t <= T:
        f.assign(ctrls[t], annotate=True)
        solve(a == L, u_0, bc)
        t += float(dt)
        adj_inc_timestep(time=t, finished=t>T)

    return u_0, list(ctrls.values())

def test_heat():
    u, ctrls = main()

    regularisation = sum([(new-old)**2 for new, old in zip(ctrls[1:], ctrls[:-1])])
    regularisation = regularisation*dx*dt_meas[START_TIME]

    alpha = Constant(1e0)
    J = Functional(u**2*dx*dt_meas + alpha*regularisation)
    m = [Control(c) for c in ctrls]

    rf = ReducedFunctional(J, m)
    minconv = rf.taylor_test(ctrls, seed=1e4)

    assert minconv > 1.9

if __name__ == "__main__":
    test_heat()
