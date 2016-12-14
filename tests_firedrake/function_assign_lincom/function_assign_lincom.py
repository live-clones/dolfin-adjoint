from firedrake import *
from firedrake_adjoint import *

def main(m):
    a = interpolate(Constant(1), m.function_space(), name="a")
    z = Function(m.function_space(), name="z")
    z.assign(0.5 * a + 2.0 * m, annotate=not parameters["adjoint"]["stop_annotating"])

    return z

if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    m = interpolate(Expression("x[0]", degree=1), V, name="m")
    z = main(m)

    adj_html("/tmp/forward.html", "forward")

    J = Functional(inner(z, z)*dx)
    m = Control(m)
    Jm = assemble(inner(z, z)*dx)

    dJ = compute_gradient(J, m, forget=False)

    def Jhat(m):
        z = main(m)
        return assemble(inner(z, z)*dx)

    minconv = taylor_test(Jhat, m, Jm, dJ)
