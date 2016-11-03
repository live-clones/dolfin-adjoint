from __future__ import print_function
from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)

def main(ic, annotate=False):
    u_prev = ic.copy(deepcopy=True, name="Temperature")
    u_next = ic.copy(deepcopy=True, name="TemperatureNext")
    u_mid  = Constant(0.5)*u_prev + Constant(0.5)*u_next

    dt = 0.001
    T = 0.1
    t = 0.0

    v = TestFunction(V)

    states = [ic.copy(deepcopy=True)]
    times  = [float(t)]

    if annotate: adj_start_timestep(time=t)
    timestep = 0

    while t < T:
        print("Solving for t == %s" % (t + dt))
        F = inner((u_next - u_prev)/Constant(dt), v)*dx + inner(grad(u_mid), grad(v))*dx
        solve(F == 0, u_next, J=derivative(F, u_next), annotate=annotate)
        u_prev.assign(u_next, annotate=annotate)

        t += dt
        timestep += 1
        states.append(u_next.copy(deepcopy=True, name="TemperatureT%s" % timestep, annotate=False))
        times.append(float(t))

        if annotate: adj_inc_timestep(time=t, finished=t>=T)

    return (times, states, u_prev)

if __name__ == "__main__":
    true_ic = interpolate(Expression("sin(2*pi*x[0])*sin(2*pi*x[1])"), V)
    (times, true_states, u) = main(true_ic, annotate=False)

    guess_ic = interpolate(Expression("15 * x[0] * (1 - x[0]) * x[1] * (1 - x[1])"), V)
    (times, computed_states, u) = main(guess_ic, annotate=True)

    combined = zip(times, true_states, computed_states)

    alpha = Constant(1.0e-7)
    J = Functional(sum(inner(true - u, true - u)*dx*dt[time] for (time, true, computed) in combined if time >= 0.01) + alpha*inner(grad(u), grad(u))*dx*dt[START_TIME])

    m = FunctionControl("Temperature")

    m_ex = Function(V, name="Temperature")
    viz  = File("output/iterations.pvd")
    def derivative_cb(j, dj, m):
        m_ex.assign(m)
        viz << m_ex

    rf = ReducedFunctional(J, m, derivative_cb_post=derivative_cb)

    problem = MinimizationProblem(rf)
    parameters = { 'maximum_iterations': 50 }

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    m_opt = problem.solve()
