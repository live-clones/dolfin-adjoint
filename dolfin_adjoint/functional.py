import libadjoint
import ufl
import ufl.algorithms
import backend
import hashlib

from . import solving
from . import adjglobals
from . import adjlinalg
from .timeforms import NoTime, StartTimeConstant, FinishTimeConstant, dt
from .timeforms import FINISH_TIME


class Functional(libadjoint.Functional):
    '''This class implements the :py:class:`libadjoint.Functional` abstract base class for dolfin-adjoint.
    The core idea is that a functional is either

      (a) an integral of a form over a certain time window, or
      (b) a pointwise evaluation in time of a certain form, or
      (c) a sum of terms like (a) and (b).

    Some examples:

      - Integration over all time:

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt)

      - Integration over a certain time window:

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt[0:1])

      - Integration from a certain point until the end:

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt[0.5:])

      - Pointwise evaluation in time (does not need to line up with timesteps):

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt[0.5])

      - Pointwise evaluation at the start (e.g. for regularisation terms):

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt[START_TIME])

      - Pointwise evaluation at the end:

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt[FINISH_TIME])

      - And sums of these work too:

        .. code-block:: python

          J = Functional(inner(u, u)*dx*dt + inner(u, u)*dx*dt[FINISH_TIME])

      If :py:data:`dt` has been redefined, you can create your own time measure with :py:class:`TimeMeasure`.

      For anything but the evaluation at the final time to work, you need to annotate the
      timestepping of your model with :py:func:`adj_inc_timestep`.'''

    def __init__(self, timeform, verbose=False, name=None):

        if isinstance(timeform, ufl.form.Form):
            if adjglobals.adjointer.adjointer.ntimesteps > 1:
                backend.info_red("You are using a steady-state functional (without the *dt term) in a time-dependent simulation.\ndolfin-adjoint will assume that you want to evaluate the functional at the end of time.")
            timeform = timeform*dt[FINISH_TIME]

            if not timeform.is_functional():
                raise Exception("Your functional must have rank 0, but is has rank >0")

        self.timeform = timeform
        self.verbose = verbose
        self.name = name

    def __add__(self, other):
        timeform = self.timeform + other.timeform
        verbose = self.verbose or other.verbose

        if self.name is None and other.name is None:
            name = None
        elif self.name is not None and other.name is None:
            name = self.name
        elif self.name is None and other.name is not None:
            name = other.name
        else:
            name = ' '.join([other.name, other.name])

        return Functional(timeform, verbose, name)

    def __mul__(self, factor):
        return Functional(factor * self.timeform, self.verbose, self.name)

    __rmul__ = __mul__

    def __div__(self, factor):
        return Functional(self.timeform / factor, self.verbose, self.name)

    # Python 3 has two div operators:
    __truediv__ = __floordiv__ = __div__

    def __neg__(self):
        return Functional(- self.timeform, self.verbose, self.name)

    def __call__(self, adjointer, timestep, dependencies, values):

        functional_value = self._substitute_form(adjointer, timestep, dependencies, values)

        if functional_value is not None:
            args = ufl.algorithms.extract_arguments(functional_value)
            if len(args) > 0:
                backend.info_red("The form passed into Functional must be rank-0 (a scalar)! You have passed in a rank-%s form." % len(args))
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs

            from .utils import _has_multimesh
            if _has_multimesh(functional_value):
                return backend.assemble_multimesh(functional_value)
            else:
                return backend.assemble(functional_value)
        else:
            return 0.0

    def derivative(self, adjointer, variable, dependencies, values):

        timer = backend.Timer("Deriving adjoint rhs symbolically")
        functional_value = None
        for timestep in self._derivative_timesteps(adjointer, variable):
            functional_value = _add(functional_value,
                                    self._substitute_form(adjointer, timestep, dependencies, values))

        if functional_value is None:
            backend.info_red("Your functional is supposed to depend on %s, but does not?" % variable)
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs

        d = backend.derivative(functional_value, values[dependencies.index(variable)].data)
        d = ufl.algorithms.expand_derivatives(d)

        if len(d.integrals()) == 0:
            raise SystemExit("This isn't supposed to happen -- your functional is supposed to depend on %s" % variable)

        print("This took {}s.".format(timer.stop()))
        return adjlinalg.Vector(d)

    def second_derivative(self, adjointer, variable, dependencies, values, contraction):

        if contraction.data is None:
            return adjlinalg.Vector(None)

        functional_value = None
        for timestep in self._derivative_timesteps(adjointer, variable):
            functional_value = _add(functional_value,
                                    self._substitute_form(adjointer, timestep, dependencies, values))

        d = backend.derivative(functional_value, values[dependencies.index(variable)].data)
        d = ufl.algorithms.expand_derivatives(d)
        d = backend.derivative(d, values[dependencies.index(variable)].data, contraction.data)
        if len(d.integrals()) == 0:
            raise SystemExit("This isn't supposed to happen -- your functional is supposed to depend on %s" % variable)
        return adjlinalg.Vector(d)

    def _derivative_timesteps(self, adjointer, variable):

        timestep = variable.timestep
        iteration = variable.iteration
        if timestep == 0 and iteration == 0:
            return [0]
        elif timestep == adjointer.timestep_count-1:
            return [timestep]
        else:
            return [timestep, timestep+1]

    def _substitute_form(self, adjointer, timestep, dependencies, values):
        ''' Perform the substitution of the dependencies and values
        provided. This is common to __call__ and __derivative__'''

        deps = {}
        for dep, val in zip(dependencies, values):
            deps[str(dep)] = val.data

        functional_value = None
        final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]

        # Get the necessary timestep information about the adjointer.
        # For integrals, we're integrating over /two/ timesteps.
        timestep_start, timestep_end = _time_levels(adjointer, timestep)
        point_interval = slice(timestep_start, timestep_end)

        if timestep > 0:
            prev_start, prev_end = _time_levels(adjointer, timestep - 1)
            integral_interval = slice(prev_start, timestep_end)
        else:
            integral_interval = slice(timestep_start, timestep_end)

        for term in self.timeform.terms:
            if isinstance(term.time, slice):
            # Integral.

                def trapezoidal(interval, iteration):
                # Evaluate the trapezoidal rule for the given interval and
                # iteration. Iteration is used to select start and end of timestep.
                    this_interval=_slice_intersect(interval, term.time)
                    if this_interval:
                        # Get adj_variables for dependencies. Time level is not yet specified.

                        # Dependency replacement dictionary.
                        replace={}

                        term_deps = _coeffs(adjointer, term.form)
                        term_vars = _vars(adjointer, term.form)

                        for term_dep, term_var in zip(term_deps,term_vars):
                            this_var = self.get_vars(adjointer, timestep, term_var)[iteration]
                            replace[term_dep] = deps[str(this_var)]

                        # Trapezoidal rule over given interval.
                        quad_weight = 0.5*(this_interval.stop-this_interval.start)

                        return backend.replace(quad_weight*term.form, replace)

                # Calculate the integral contribution from the previous time level.
                functional_value = _add(functional_value, trapezoidal(integral_interval, 0))

                # On the final occasion, also calculate the contribution from the
                # current time level.
                if adjointer.finished and timestep == adjointer.timestep_count - 1: # we're at the end, and need to add the extra terms
                                                                                    # associated with that
                    final_interval = slice(timestep_start, timestep_end)
                    functional_value = _add(functional_value, trapezoidal(final_interval, 1))

            else:
                # Point evaluation.

                if point_interval.start < term.time < point_interval.stop:
                    replace = {}

                    term_deps = _coeffs(adjointer, term.form)
                    term_vars = _vars(adjointer, term.form)

                    for term_dep, term_var in zip(term_deps, term_vars):
                        (start, end) = self.get_vars(adjointer, timestep, term_var)
                        theta = float(1.0 - (term.time - point_interval.start)/(point_interval.stop - point_interval.start))
                        replace[term_dep] = theta*deps[str(start)] + (1-theta)*deps[str(end)]

                    functional_value = _add(functional_value, backend.replace(term.form, replace))

                # Special case for evaluation at the end of time: we can't pass over to the
                # right-hand timestep, so have to do it here.
                elif (term.time == final_time or isinstance(term.time, FinishTimeConstant)) and point_interval.stop == final_time:
                    replace = {}

                    term_deps = _coeffs(adjointer, term.form)
                    term_vars = _vars(adjointer, term.form)

                    for term_dep, term_var in zip(term_deps, term_vars):
                        end = self.get_vars(adjointer, timestep, term_var)[1]
                        replace[term_dep] = deps[str(end)]

                    functional_value = _add(functional_value, backend.replace(term.form, replace))

                # Another special case for the start of a timestep.
                elif (isinstance(term.time, StartTimeConstant) and timestep == 0) or point_interval.start == term.time:
                    replace = {}

                    term_deps = _coeffs(adjointer, term.form)
                    term_vars = _vars(adjointer, term.form)

                    for term_dep, term_var in zip(term_deps, term_vars):
                        end = self.get_vars(adjointer, timestep, term_var)[0]
                        replace[term_dep] = deps[str(end)]

                    functional_value = _add(functional_value, backend.replace(term.form, replace))

        return functional_value

    def get_vars(self, adjointer, timestep, model):
        # Using the adjointer, get the start and end variables associated
        # with this timestep
        start = model.copy()
        end = model.copy()

        if timestep == 0:
            start.timestep = 0
            start.iteration = 0
            end.timestep = 0
            end.iteration = end.iteration_count(adjointer) - 1
        else:
            try:
                start.timestep = timestep - 1
                start.iteration = start.iteration_count(adjointer) - 1
            except libadjoint.exceptions.LibadjointErrorInvalidInputs:
                start = None

            end.timestep = timestep
            end.iteration = end.iteration_count(adjointer) - 1

        return (start, end)

    def get_form(self, adjointer, timestep):
        deps = self.dependencies(adjointer, timestep)
        values = [adjointer.get_variable_value(dep) for dep in deps]
        form = self._substitute_form(adjointer, timestep, deps, values)
        return form

    def dependencies(self, adjointer, timestep):

        point_deps = set()
        integral_deps = set()
        final_deps = set()
        start_deps = set()

        levels = _time_levels(adjointer, timestep)
        point_interval=slice(levels[0],levels[1])
        final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]

        prev_levels = _time_levels(adjointer, max(timestep-1,0))
        integral_interval=slice(prev_levels[0],levels[1])

        for term in self.timeform.terms:
            if isinstance(term.time, slice):
                # Integral.

                # Get adj_variables for dependencies. Time level is not yet specified.

                if _slice_intersect(integral_interval, term.time):
                    integral_deps.update(_vars(adjointer, term.form))

            else:

                # Point evaluation.
                if point_interval.start < term.time < point_interval.stop:
                    point_deps.update(_vars(adjointer, term.form))

                # Special case for evaluation at the end of time: we can't pass over to the
                # right-hand timestep, so have to do it here.
                elif (term.time == final_time or isinstance(term.time, FinishTimeConstant)) and point_interval.stop == final_time:
                    final_deps.update(_vars(adjointer, term.form))

                # Another special case for evaluation at the START_TIME, or the start of the timestep.
                elif (isinstance(term.time, StartTimeConstant) and timestep == 0) or point_interval.start == term.time:
                    start_deps.update(_vars(adjointer, term.form))

        integral_deps = list(integral_deps)
        point_deps = list(point_deps)
        final_deps = list(final_deps)
        start_deps = list(start_deps)

        # Set the time level of the dependencies:

        # Point deps always need the current and previous timestep values.
        point_deps *= 2
        for i in range(len(point_deps)//2):
            point_deps[i]= point_deps[i].copy()
            if timestep !=0:
                point_deps[i].var.timestep = timestep-1
                point_deps[i].var.iteration = point_deps[i].iteration_count(adjointer) - 1
            else:
                point_deps[i].var.timestep = timestep
                point_deps[i].var.iteration = 0
        for i in range(len(point_deps)//2, len(point_deps)):
            point_deps[i].var.timestep = timestep
            point_deps[i].var.iteration = point_deps[i].iteration_count(adjointer) - 1

        # Integral deps depend on the previous time level.
        for i in range(len(integral_deps)):
            if timestep !=0:
                integral_deps[i].var.timestep = timestep - 1
                integral_deps[i].var.iteration = integral_deps[i].iteration_count(adjointer) - 1
            else:
                integral_deps[i].var.timestep = timestep
                integral_deps[i].var.iteration = 0

        # Except at the final timestep, integrals only depend on the previous
        # value.
        if  timestep==adjointer.timestep_count-1 and adjointer.time.finished:
            integral_deps*=2
            for i in range(len(integral_deps)//2, len(integral_deps)):
                integral_deps[i]= integral_deps[i].copy()
                integral_deps[i].var.timestep = timestep
                integral_deps[i].var.iteration = integral_deps[i].iteration_count(adjointer) - 1

        # Final deps depend only on the very last value.
        for i in range(len(final_deps)):
            final_deps[i] = self.get_vars(adjointer, timestep, final_deps[i])[1]

        # Start deps depend only at the very first value.
        for i in range(len(start_deps)):
            start_deps[i] = self.get_vars(adjointer, timestep, start_deps[i])[0]

        deps=set(point_deps).union(set(integral_deps)).union(set(final_deps)).union(set(start_deps))

        return list(deps)

    def __str__(self):
        if self.name is not None:
            return "Functional:" + self.name
        else:
            formbytes = " ".join([str(hash(term)) for term in self.timeform.terms]).encode('utf8')
            return "Functional:" + hashlib.md5(formbytes).hexdigest()

def _slice_intersect(slice1, slice2):

    if slice1.stop>=slice2.start and slice2.stop>slice1.start:
        intersect=slice(max(slice1.start, slice2.start),min(slice1.stop,slice2.stop))
        if intersect.start==intersect.stop:
            # A zero length intersect doesn't count.
            return None
        else:
            return intersect
    else:
        return None

def _vars(adjointer, form):
    # Return the libadjoint variables corresponding to the coeffs in form.
    return [adjglobals.adj_variables[coeff].copy()
            for coeff in ufl.algorithms.extract_coefficients(form)
            if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]

def _coeffs(adjointer, form):
    return [coeff
            for coeff in ufl.algorithms.extract_coefficients(form)
            if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]

def _add(value, increment):
    # Add increment to value correctly taking into account None.
    if increment is None:
        return value
    elif value is None:
        return increment
    else:
        return value+increment

def _time_levels(adjointer, timestep):

    try:
        return adjointer.get_times(timestep)
    except Exception as exc:
        start = NoTime(str(exc))
        end = NoTime(str(exc))

        if timestep == adjointer.timestep_count - 1:
            end = FinishTimeConstant()

        if timestep == 0:
            start = StartTimeConstant()

        return (start, end)
