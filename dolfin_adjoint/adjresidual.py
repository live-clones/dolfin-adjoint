from . import adjlinalg
from . import adjglobals

import backend
import ufl
import ufl.algorithms

import libadjoint

from backend import info_red, info_blue

def get_residual(i):
    from .adjrhs import adj_get_forward_equation
    (fwd_var, lhs, rhs) = adj_get_forward_equation(i)

    if isinstance(lhs, adjlinalg.IdentityMatrix):
        return None

    fn_space = ufl.algorithms.extract_arguments(lhs)[0].function_space()
    x = backend.Function(fn_space)

    if rhs == 0:
        form = lhs
        x = fwd_var.nonlinear_u
    else:
        form = backend.action(lhs, x) - rhs

    try:
        y = adjglobals.adjointer.get_variable_value(fwd_var).data
    except libadjoint.exceptions.LibadjointErrorNeedValue:
        info_red("Warning: recomputing forward solution; please report this script on launchpad")
        y = adjglobals.adjointer.get_forward_solution(i)[1].data

    form = backend.replace(form, {x: y})

    return form
