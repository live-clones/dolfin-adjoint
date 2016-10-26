import sys

import kelvin_new as kelvin
import sw_lib

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"]=True

mesh = UnitSquareMesh(6, 6)
W=sw_lib.p1dgp2(mesh)

state=Function(W)

state.interpolate(kelvin.InitialConditions(degree=1))

kelvin.params["basename"] = "p1dgp2"
kelvin.params["dt"] = 2
kelvin.params["finish_time"] = kelvin.params["dt"]*2
kelvin.params["dump_period"] = 1

M, G=sw_lib.construct_shallow_water(W, kelvin.params)

state = sw_lib.timeloop_theta(M, G, state, kelvin.params)

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")

replay_dolfin()
J = Functional(dot(state, state)*dx*dt[FINISH_TIME])
f_direct = assemble(dot(state, state)*dx)
for (adj_state, var) in compute_adjoint(J):
    pass

ic = Function(W)
ic.interpolate(kelvin.InitialConditions(degree=1))
def compute_J(ic):
    state = sw_lib.timeloop_theta(M, G, ic, kelvin.params, annotate=False)
    return assemble(dot(state, state)*dx)

minconv = utils.test_initial_condition_adjoint(compute_J, ic, adj_state, seed=0.001)
if minconv < 1.9:
    exit_code = 1
else:
    exit_code = 0
sys.exit(exit_code)
