# Author: Steven Vandekerckhove <Steven.Vandekerckhove@kuleuven.be>

from __future__ import print_function
import libadjoint
from ufl import *
import ufl.algorithms
import backend
import numpy as np

import dolfin_adjoint.functional as functional
from dolfin_adjoint.functional import _time_levels, _add, _coeffs, _vars
from dolfin_adjoint.timeforms import dt
import dolfin_adjoint.adjlinalg as adjlinalg

class PointwiseFunctional(functional.Functional):
    '''The Functional class is overloaded to handle specific functionals,
    where a Functions is evaluated in one or multiple points.

    Args:
        u (Function): The computed solution.
        refs (List of lists or scalars): The reference.
        coords (List of Points): The coordinates of the points where u is evaluated in the functional
        times (List of scalars): The time instances considered in the functional
          Default: None (assuming final time)
        u_ind (List of ints): In case u is a vector function, u_ind contains a list of which component of u that is studied
          Default: [None] (assuming a scalar function u)
        Verbose (bool): Show info on the commandline while running
          Default: False
        Name (str): Name of the functional
        regularisation (Form): the regularisation term (will be evaluated at START_TIME)
        alpha (float): Constant to multiply the misfit terms with

    Limitations:

        - This code need further development to run in parallel.
        - Taylor checks are not yet correct when using a regularisation term.


    '''

    #-----------------------------------------------------------------------------------------------------
    def __init__(self, u, refs, coords, times=None, **kwargs):

        # Sort out input params
        self.coords   = coords    # List of Points
        self.func     = u         # Dolfin function to be evaluated
        self.refs     = refs      # References
        self.times    = times     # Relevant times
        self.timeform = kwargs.get("timeform", False)
        self.verbose  = kwargs.get("verbose",  False)
        self.name     = kwargs.get("name", None)
        self.regform  = kwargs.get("regularisation", None)
        self.alpha    = kwargs.get("alpha", 1.0)
        self.index    = kwargs.get("u_ind", [None])

        # Prep coords to be considerd as a matrix
        if type(self.coords) is not list:
            self.coords = [self.coords]
            self.refs = [self.refs]

        self.basis    = [None]*len(self.coords)
        self.skip     = [False]*len(self.coords)

        # Some conformity checks
        if self.times is None:
            self.times = ["FINISH_TIME"]
        elif len(self.times) < 1:
            raise RuntimeError("""The 'times' argument should be None,
                                    'FINISH_TIME' or a non-empty list""")


        if len(self.index) != len(self.coords):
            raise RuntimeError("""The 'index' argument should be of the,
                                same length as the 'coords argument'""")

        # Prepare a ghost timeform. Only the time instant is important.
        self.timeform = sum(inner(u,u)*dx*dt[t] for t in self.times)

        # Add regularisation
        if self.regform is not None:
            self.timeform += self.regform*dt[0]
            self.regfunc  = functional.Functional(self.regform*dt[0])

        # check compatibility inputs
        if len(self.coords) != len(self.refs):
            raise RuntimeError("Number of coordinates and observations doesn't match %4i vs %4i" %(len(self.coords), len(self.refs)))
        else:
            for self.ref in self.refs:
              if len(self.ref) != len(self.times): # check compatibility inputs
                raise RuntimeError("Number of timesteps and observations doesn't match %4i vs %4i" %(len(self.times), len(self.ref)))

        for i in range(len(self.coords)):
            # Prepare pointwise evals for derivative
            if self.index[i] is None:
                ps = backend.PointSource(self.func.function_space(), self.coords[i], 1.)
            else:
                ps = backend.PointSource(self.func.function_space().sub(self.index[i]), self.coords[i], 1.)
            self.basis[i] = backend.Function(self.func.function_space()) # basis function for R
            ps.apply(self.basis[i].vector())

            # Failsaife for parallel
            if sum(self.basis[i].vector().array())<1.e-12:
                if self.verbose: print("coord %i not in domain" %i)
                self.skip[i] = True

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional
    def __call__(self, adjointer, timestep, dependencies, values):

        if self.verbose:
            print("eval ", len(values))
            print("timestep ", timestep)
            print("\r\n******************")

        toi = _time_levels(adjointer, timestep)[0] # time of interest

        my   = [0.0]*len(self.coords)
        for i in range(len(self.coords)):
            if not self.skip[i] and len(values) > 0:
                if timestep is adjointer.timestep_count -1:

                    # add final contribution
                    if self.index[i] is None: solu = values[0].data(self.coords[i])
                    else: solu = values[0].data(self.coords[i])[self.index[i]]
                    ref  = self.refs[i][self.times.index(self.times[-1])]
                    my[i] = (solu - float(ref))*(solu - float(ref))

                    # if necessary, add one but last contribution
                    if toi in self.times and len(values) > 0:
                        if self.index[i] is None: solu = values[-1].data(self.coords[i])
                        else: solu = values[-1].data(self.coords[i])[self.index[i]]
                        ref  = self.refs[i][self.times.index(toi)]
                        my[i] += (solu - float(ref))*(solu - float(ref))

                elif timestep is 0:
                    return backend.assemble(self.regform)
                else: # normal situation
                    if self.index[i] is None: solu = values[-1].data(self.coords[i])
                    else: solu = values[-1].data(self.coords[i])[self.index[i]]
                    ref  = self.refs[i][self.times.index(toi)]
                    my[i] = (solu - float(ref))*(solu - float(ref))

            if self.verbose:
                print("my eval ", my[i])
                print("eval ", timestep, " times ", _time_levels(adjointer, timestep))

        return self.alpha*sum(my)

    #-----------------------------------------------------------------------------------------------------
    # Evaluate functional derivative
    def derivative(self, adjointer, variable, dependencies, values):
        if self.verbose:
            for dep in dependencies:
                print(variable.timestep, "derive wrt ", dep.name)
        if self.regform is not None and variable.name == self.regform.coefficients()[0].name(): # derivative wrt the contorl
            if self.verbose: " derivatives wrt the controls "
            raise RuntimeError("""The derivative of a regularisation term
                                  doesn't work properly and shouldn't be used""")
            return self.regfunc.derivative(adjointer, variable, dependencies, values)
        else:
            # transate finish_time: UGLY!!
            if "FINISH_TIME" in self.times:
                final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]
                self.times[self.times.index("FINISH_TIME")] = final_time

            if self.verbose: print("derive ", variable.timestep, " num values ", len(values))
            timesteps = self._derivative_timesteps(adjointer, variable)

            ff    = [backend.Constant(0.0)]*len(self.coords)
            for i in range(len(self.coords)):
                if self.skip[i]:
                    if self.verbose: print("skipped")
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
                    if self.index[i] is None: solu = coef(self.coords[i])
                    else: solu = coef(self.coords[i])[self.index[i]]
                    ff[i] = backend.Constant(self.alpha*2.0*(solu - float(ref)))

            # Set up linear combinations to be projected
            form = ff[0]*self.basis[0]
            for i in range(1, len(self.coords)): form += ff[i]*self.basis[i]

            v = backend.project(form, self.func.function_space())

            return adjlinalg.Vector(v)
