#!/usr/bin/env python3

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014, 2017 University of Edinburgh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Modified version of shallow_water test from dolfin-adjoint bzr trunk 636
# Code first added: 2013-04-26

from __future__ import print_function
from dolfin import *
import numpy
import sys
import six

class parameters(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''
    def __init__(self, dict={}):
        self["current_time"]=0.0
        self["theta"]=0.5

        # Apply dict after defaults so as to overwrite the defaults
        for key,val in six.iteritems(dict):
            self[key]=val

        self.required={
            "depth" : "water depth",
            "dt" : "timestep",
            "finish_time" : "finish time",
            "dump_period" : "dump period in timesteps",
            "basename" : "base name for I/O"
            }

    def check(self):
        for key, error in six.iteritems(self.required):
            if key not in self:
                sys.stderr.write("Missing parameter: "+key+"\n"+
                                 "This is used to set the "+error+"\n")
                raise KeyError

def rt0(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = FunctionSpace(mesh, 'Raviart-Thomas', 1) # Velocity space

    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W=V*H                                        # Mixed space of both.

    return W
def p1dgp2(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = VectorFunctionSpace(mesh, 'DG', 1, dim=2)# Velocity space

    H = FunctionSpace(mesh, 'CG', 2)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdfmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDFM', 1)# Velocity space

    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdmp0(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)# Velocity space

    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)# Velocity space

    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def construct_shallow_water(W,params):
    """Construct the linear shallow water equations for the space W(=U*H) and a
    dictionary of parameters params."""
    # Sanity check for parameters.
    params.check()

    (v, q)=TestFunctions(W)
    (u, h)=TrialFunctions(W)

    n=FacetNormal(W.mesh())

    # Mass matrix
    M=inner(v,u)*dx
    M+=inner(q,h)*dx

    # Divergence term.
    Ct=-inner(u,grad(q))*dx+inner(avg(u),jump(q,n))*dS
    #Ct=div(u)*q*dx
    # Pressure gradient operator
    C=(params["g"]*params["depth"])*\
        inner(v,grad(h))*dx+inner(avg(v),jump(h,n))*dS

    try:
        # Coriolis term
        F=params["f"]*inner(v,as_vector([-u[1],u[0]]))*dx
    except KeyError:
        F=0

    if "big_spring" in params:
        print("big spring active: ", params["big_spring"])
        C+=inner(v,n)*inner(u,n)*params["big_spring"]*ds

    return (M, C+Ct+F)

def add_timeloop_solve(system, State, W, params):
    from timestepping import n

    test = TestFunction(W)
    v, q = split(test)
    nm = FacetNormal(W.mesh())

    def rhs(state):
        u, h = split(state)

        # Divergence term
        L = -inner(grad(q), u) * dx + inner(jump(q, nm), avg(u)) * dS
        # Pressure gradient term
        L += params["g"] * params["depth"] * inner(v, grad(h)) * dx + inner(avg(v), jump(h, nm)) * dS
        if "f" in params:
            # Coriolis term
            L += params["f"] * inner(v, as_vector([-u[1], u[0]])) * dx
        if "big_spring" in params:
            L += inner(v, nm) * inner(u, nm) * params["big_spring"] * ds
        return params["dt"] * -L

    system.add_solve(inner(test, State[n + 1] - State[n]) * dx ==
      params["theta"] * rhs(State[n + 1]) + (1.0 - params["theta"]) * rhs(State[n]),
      State[n + 1])

    return

def timeloop_theta(M, G, state, params, annotate=True):
    '''Solve M*dstate/dt = G*state using a theta scheme.'''

    A=M+params["theta"]*params["dt"]*G

    A_r=M-(1-params["theta"])*params["dt"]*G

    u_out,p_out=output_files(params["basename"])

    M_u_out, v_out, u_out_state=u_output_projector(state.function_space())

    M_p_out, q_out, p_out_state=p_output_projector(state.function_space())

    # Project the solution to P1 for visualisation.
    rhs=assemble(inner(v_out,state.split()[0])*dx)
    solve(M_u_out, u_out_state.vector(),rhs,"cg","sor", annotate=False)

    # Project the solution to P1 for visualisation.
    rhs=assemble(inner(q_out,state.split()[1])*dx)
    solve(M_p_out, p_out_state.vector(),rhs,"cg","sor", annotate=False)

    u_out << u_out_state
    p_out << p_out_state

    t = params["current_time"]
    dt= params["dt"]

    step=0

    tmpstate=Function(state.function_space())

    while (t < params["finish_time"]):
        t+=dt
        step+=1
        rhs=action(A_r,state)

        # Solve the shallow water equations.
        solve(A==rhs, tmpstate, annotate=annotate)
        #solve(A, state.vector(), rhs, "preonly", "lu")

        state.assign(tmpstate, annotate=annotate)

        if step%params["dump_period"] == 0:

            # Project the solution to P1 for visualisation.
            rhs=assemble(inner(v_out,state.split()[0])*dx)
            solve(M_u_out, u_out_state.vector(),rhs,"cg","sor", annotate=False)

            # Project the solution to P1 for visualisation.
            rhs=assemble(inner(q_out,state.split()[1])*dx)
            solve(M_p_out, p_out_state.vector(),rhs,"cg","sor", annotate=False)

            u_out << u_out_state
            p_out << p_out_state

    return state # return the state at the final time

def replay(state,params):

    print("Replaying forward run")

    for i in range(adjointer.equation_count):
        (fwd_var, output) = adjointer.get_forward_solution(i)

        s=libadjoint.MemoryStorage(output)
        s.set_compare(0.0)
        s.set_overwrite(True)

        adjointer.record_variable(fwd_var, s)

def adjoint(state, params, functional):

    print("Running adjoint")

    for i in range(adjointer.equation_count)[::-1]:
        print("  solving adjoint equation ", i)
        (adj_var, output) = adjointer.get_adjoint_solution(i, functional)

        s=libadjoint.MemoryStorage(output)
        adjointer.record_variable(adj_var, s)

    return output.data # return the adjoint solution associated with the initial condition


def u_output_projector(W):
    # Projection operator for output.
    Output_V=VectorFunctionSpace(W.mesh(), 'CG', 1, dim=2)

    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)

    M_out=assemble(inner(v_out,u_out)*dx)

    out_state=Function(Output_V)

    return M_out, v_out, out_state

def p_output_projector(W):
    # Projection operator for output.
    Output_V=FunctionSpace(W.mesh(), 'CG', 1)

    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)

    M_out=assemble(inner(v_out,u_out)*dx)

    out_state=Function(Output_V)

    return M_out, v_out, out_state

def output_files(basename):

    # Output file
    u_out = File(basename+"_u.pvd", "compressed")
    p_out = File(basename+"_p.pvd", "compressed")

    return u_out, p_out
