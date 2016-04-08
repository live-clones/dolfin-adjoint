from dolfin import *
from dolfin_adjoint import *

import random
import libadjoint
import sys

mesh = UnitSquareMesh(2, 2)
V2 = FunctionSpace(mesh, "CG", 2)
V1 = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V2, V1])

def main(ic, fnsplit=True, annotate=False):
    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    if fnsplit:
        ic_u = ic.split()[0]
    else:
        ic_u = split(ic)[0]

    mass = inner(w, v)*dx
    rhs  = inner(ic_u, v)*dx

    solve(mass == rhs, u, annotate=annotate)

    return u

if __name__ == "__main__":
    ic = Function(Z)
    vec = ic.vector()
    for i in range(len(vec)):
        vec[i] = random.random()

    raised_exception = False
    try:
        u = main(ic, fnsplit=True, annotate=True)
    except libadjoint.exceptions.LibadjointErrorNotImplemented:
        raised_exception = True

    if not raised_exception:
        sys.exit(1)

    raised_exception = False
    try:
        u = main(ic, fnsplit=False, annotate=True)
    except:
        raised_exception = True

    if raised_exception:
        sys.exit(1)
