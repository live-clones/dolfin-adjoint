# Author: Steven Vandekerckhove <Steven.Vandekerckhove@kuleuven.be>

import libadjoint
from ufl import *
import ufl.algorithms
import backend
import hashlib
from IPython import embed as key

import dolfin_adjoint.functional as functional
from dolfin_adjoint.functional import _time_levels, _add, _coeffs, _vars
from dolfin_adjoint.timeforms import dt
import dolfin_adjoint.adjlinalg as adjlinalg

from IPython import embed

class PointwiseFunctional(functional.Functional):
    # u : function containing the solution
    # refs : vector containing the observations
    # coord : coordinates of the point in which the functional has to be evaluated
    # times: list of the time instants that are considered in the functional
    # f_ind: the inde of the component of u that is considered
    # timeform: ??
    # verbose : T/F Spill out info while running
    # name: allow to set a name, which is usefull for all da obejcts

    #-----------------------------------------------------------------------------------------------------
    def __init__(self, u, refs, coords, times=None, f_ind=None, timeform=False, verbose=False, name=None):
        if times is None:
            times = ["FINISH_TIME"]
        elif len(times) < 1:
            raise RuntimeError("""The 'times' argument should be None,
                                    'FINISH_TIME' or a non-empty list""")

        # we prepare a ghost timeform. Only the time instant is important
        if not timeform:
            if f_ind is None:
                self.timeform = sum(u*dx*dt[t] for t in times)
            else:
                self.timeform = sum(u[f_ind]*dx*dt[t] for t in times)
        else: self.timeform = timeform

        # Store info to object
        self.coords = coords    # Numpy array
        self.verbose = verbose
        self.name = name
        self.func = u
        self.refs = refs
        self.times = times
        self.i = f_ind
        self.boost = 1e20

        # check compatibility inputs
        if len(u) is not len(times):
            raise RuntimeError("Number of timesteps and observations don't match")

        # Prepare pointwise evals for derivative
        ps = backend.PointSource(self.func.function_space().sub(self.i), backend.Point(self.coords), 1.)
        self.basis = backend.Function(self.func.function_space()) # basis function for R
        ps.apply(self.basis.vector())

        if self.verbose: print "basis = ", self.basis

        # Failsaife for parallel
        if sum(self.basis.vector().array())<1.:
            if self.verbose: print "coord not in domain"
            self.skip = True
        else:
            self.skip = False

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional
    def __call__(self, adjointer, timestep, dependencies, values):
        if self.verbose: print "eval ", len(values)
        toi = _time_levels(adjointer, timestep)[0] # time of interest
        if not self.skip and len(values) > 0:
            if timestep is adjointer.timestep_count -1:

                # add final contribution
                if self.i is None: solu = values[0].data(self.coords)
                else: solu = values[0].data[self.i](self.coords)
                ref  = self.refs[self.times.index(self.times[-1])]
                my = (solu - float(ref))*(solu - float(ref))

                # if necessary, add one but last contribution
                if toi in self.times and len(values) > 0:
                    if self.i is None: solu = values[-1].data(self.coords)
                    else: solu = values[-1].data[self.i](self.coords)
                    ref  = self.refs[self.times.index(toi)]
                    my += (solu - float(ref))*(solu - float(ref))
            else:
                if self.i is None: solu = values[-1].data(self.coords)
                else: solu = values[-1].data[self.i](self.coords)
                ref  = self.refs[self.times.index(toi)]
                my = (solu - float(ref))*(solu - float(ref))
        else:
            my = 0.0

        if self.verbose: print "my eval ", my
        if self.verbose:print "eval ", timestep, " times ", _time_levels(adjointer, timestep)

        return self.boost*my

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional derivative
    def derivative(self, adjointer, variable, dependencies, values):
        # transate finish_time: UGLY!!
        if "FINISH_TIME" in self.times:
            final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]
            self.times[self.times.index("FINISH_TIME")] = final_time

        if self.verbose:print "derive ", variable.timestep, " num values ", len(values)
        timesteps = self._derivative_timesteps(adjointer, variable)
        if self.skip:
            if self.verbose: print "skipped"
            v = self.basis
        else:
            if len(timesteps) is 1: # only occurs at start and finish time
                tsoi = timesteps[-1]
                if tsoi is 0: toi = _time_levels(adjointer, tsoi)[0]; ind = -1
                else: toi = _time_levels(adjointer, tsoi)[-1]; ind = 0
            else:
                if len(values) is 1: # one value (easy)
                    tsoi = timesteps[-1]
                    toi = _time_levels(adjointer, tsoi)[0]
                    ind = 0
                elif len(values) is 2: # two values (hard)
                    tsoi = timesteps[-1]
                    toi = _time_levels(adjointer, tsoi)[0]
                    if _time_levels(adjointer, tsoi)[1] in self.times: ind = 0
                    else: ind = 1
                else: # three values (easy)
                    tsoi = timesteps[1]
                    toi = _time_levels(adjointer, tsoi)[0]
                    ind = 1
            coef = values[ind].data
            ref  = self.refs[self.times.index(toi)]
            if self.i is None: solu = coef(self.coords)
            else: solu = coef[self.i](self.coords)
            ff = backend.Constant(self.boost*2.0*(solu - float(ref)))
            v = backend.project(ff*self.basis, self.func.function_space())

            if self.verbose: print "ff", float(ff)
            if self.verbose: print "sol", solu
            if self.verbose: print "ref", float(ref)

            if self.verbose: print "tsoi", tsoi
            if self.verbose: print "toi", toi

        my = v.vector().norm("l2")
        if self.verbose: print "my", my

        return adjlinalg.Vector(v)
