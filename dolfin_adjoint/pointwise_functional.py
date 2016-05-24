# Author: Steven Vandekerckhove <Steven.Vandekerckhove@kuleuven.be>

import libadjoint
from ufl import *
import ufl.algorithms
import backend
import hashlib
from IPython import embed as key
import numpy as np

import dolfin_adjoint.functional as functional
from dolfin_adjoint.functional import _time_levels, _add, _coeffs, _vars
from dolfin_adjoint.timeforms import dt
import dolfin_adjoint.adjlinalg as adjlinalg

class PointwiseFunctional(functional.Functional):
    # u : function containing the solution
    # refs : vector containing the observations
    # coord : coordinates of the point in which the functional has to be evaluated
    # times: list of the time instants that are considered in the functional
    # u_ind: the inde of the component of u that is considered
    # timeform: ??
    # verbose : T/F Spill out info while running
    # name: allow to set a name, which is usefull for all da obejcts

    #-----------------------------------------------------------------------------------------------------
    def __init__(self, u, refs, coords, times=None, **kwargs):

        # Sort out input params
        self.coords   = coords    # Numpy array with coordinates
        self.func     = u         # Dolfin function to be evaluated
        self.refs     = refs      # References
        self.times    = times     # Relevant times
        self.timeform = kwargs.get("timeform", False)
        self.verbose  = kwargs.get("verbose",  False)
        self.name     = kwargs.get("name", None)
        self.regform  = kwargs.get("regform", None)
        self.boost    = kwargs.get("boost", 1.0)
        self.index    = kwargs.get("u_ind", [None])
        self.basis    = [None]*self.coords.shape[0]
        self.skip     = [False]*self.coords.shape[0]

        # Some conformity checks
        if self.times is None:
            self.times = ["FINISH_TIME"]
        elif len(self.times) < 1:
            raise RuntimeError("""The 'times' argument should be None,
                                    'FINISH_TIME' or a non-empty list""")

        if self.coords.shape[0] > 1:
            if len(self.index) != self.coords.shape[0]:
                raise RuntimeError("""The 'index' argument should be of the,
                                    same length as the 'coords argument'""")

        # Prep coords to be considerd as a matrix
        if self.coords.ndim == 1:
            self.coords = np.array([self.coords])
            self.refs = [self.refs]

        # Prepare a ghost timeform. Only the time instant is important.
        if not self.timeform:
            self.timeform = sum(inner(u,u)*dx*dt[t] for t in self.times)

        # Add regularisation
        if self.regform is not None:
            self.timeform += self.regform*dt[0]
            self.regfunc  = functional.Functional(self.regform*dt[0])

        # check compatibility inputs
        if self.coords.shape[0] != len(self.refs):
            raise RuntimeError("Number of coordinates and observations doesn't match %4i vs %4i" %(self.coords.shape[0], len(self.refs)))
        else:
            for self.ref in self.refs:
              if len(self.ref) != len(self.times): # check compatibility inputs
                raise RuntimeError("Number of timesteps and observations doesn't match %4i vs %4i" %(len(self.times), len(self.refs)))

        for i in range (self.coords.shape[0]):
            # Prepare pointwise evals for derivative
            if self.index[i] is None:
                ps = backend.PointSource(self.func.function_space(), backend.Point(self.coords[i,:]), 1.)
            else:
                ps = backend.PointSource(self.func.function_space().sub(self.index[i]), backend.Point(self.coords[i,:]), 1.)
            self.basis[i] = backend.Function(self.func.function_space()) # basis function for R
            ps.apply(self.basis[i].vector())

            # Failsaife for parallel
            if sum(self.basis[i].vector().array())<1.e-12:
                if self.verbose: print "coord %i not in domain" %i
                self.skip[i] = True

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional
    def __call__(self, adjointer, timestep, dependencies, values):

        if self.verbose: print "eval ", len(values)
        print "\r\n******************"
        toi = _time_levels(adjointer, timestep)[0] # time of interest

        my   = [0.0]*self.coords.shape[0]
        for i in range (self.coords.shape[0]):
            if not self.skip[i] and len(values) > 0:
                if timestep is adjointer.timestep_count -1:

                    # add final contribution
                    if self.index[i] is None: solu = values[0].data(self.coords[i,:])
                    else: solu = values[0].data[self.index[i]](self.coords[i,:])
                    ref  = self.refs[i][self.times.index(self.times[-1])]
                    my[i] = (solu - float(ref))*(solu - float(ref))

                    if self.verbose:
                        print "add final contrib"
                        print ref, " ", float(ref)
                        print ref, " ", solu

                    # if necessary, add one but last contribution
                    if toi in self.times and len(values) > 0:
                        if self.index[i] is None: solu = values[-1].data(self.coords[i,:])
                        else: solu = values[-1].data[self.index[i]](self.coords[i,:])
                        ref  = self.refs[i][self.times.index(toi)]
                        my[i] += (solu - float(ref))*(solu - float(ref))

                        if self.verbose:
                            print "add contrib"
                            print ref, " ", float(ref)
                            print ref, " ", solu
                elif timestep is 0:
                    return backend.assemble(self.regform)
                else:
                    if self.index[i] is None: solu = values[-1].data(self.coords[i,:])
                    else: solu = values[-1].data[self.index[i]](self.coords[i,:])
                    ref  = self.refs[i][self.times.index(toi)]
                    my[i] = (solu - float(ref))*(solu - float(ref))

                    if self.verbose:
                        print "add regular contrib"
                        print ref, " ", float(ref)
                        print ref, " ", solu

            if self.verbose:
                print "my eval ", my[i]
                print "eval ", timestep, " times ", _time_levels(adjointer, timestep)

        return self.boost*sum(my)

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional derivative
    def derivative(self, adjointer, variable, dependencies, values):
        for dep in dependencies: print variable.timestep, "derive wrt ", dep.name

        if variable.timestep is 0 and self.regform is not None:
            " derivatives wrt the controls "
            d = derivative(self.regform, self.regform.coefficients()[0])
            return self.regfunc.derivative(adjointer, variable, dependencies, values)

        # transate finish_time: UGLY!!
        if "FINISH_TIME" in self.times:
            final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]
            self.times[self.times.index("FINISH_TIME")] = final_time

        if self.verbose:print "derive ", variable.timestep, " num values ", len(values)
        timesteps = self._derivative_timesteps(adjointer, variable)

        ff    = [0.0]*self.coords.shape[0]
        for i in range (self.coords.shape[0]):
            if self.skip[i]:
                if self.verbose: print "skipped"
                v[i] = self.basis[i]
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
                ref  = self.refs[i][self.times.index(toi)]
                if self.index[i] is None: solu = coef(self.coords[i,:])
                else: solu = coef[self.index[i]](self.coords[i,:])
                ff[i] = backend.Constant(self.boost*2.0*(solu - float(ref)))

                if self.verbose:
                    print "ff", float(ff[i])
                    print "sol", solu
                    print "ref", float(ref)
                    print "tsoi", tsoi
                    print "toi", toi

        # Set up linear combinations to be projected
        form = ff[0]*self.basis[0]
        for i in range(1, self.coords.shape[0]): form += ff[i]*self.basis[i]

        v = backend.project(form, self.func.function_space())

        return adjlinalg.Vector(v)
