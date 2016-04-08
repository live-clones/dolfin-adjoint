import ufl
import libadjoint

class TimeConstant(object):
    def __init__(self, label):
        self.label = label
    def __repr__(self):
        return 'TimeConstant("'+self.label+'")'

class StartTimeConstant(TimeConstant):
    def __init__(self):
        TimeConstant.__init__(self, "START_TIME")
    def __cmp__(self, other):
        if isinstance(other, StartTimeConstant):
            return 0
        return -1
    def __repr__(self):
        return "StartTimeConstant()"

class FinishTimeConstant(TimeConstant):
    def __init__(self):
        TimeConstant.__init__(self, "FINISH_TIME")
    def __cmp__(self, other):
        if isinstance(other, FinishTimeConstant):
            return 0
        return 1
    def __repr__(self):
        return "FinishTimeConstant()"

class NoTime(TimeConstant):
    def __init__(self, msg=""):
        TimeConstant.__init__(self, msg)
    def __cmp__(self, other):
        if isinstance(other, StartTimeConstant):
            return 1
        elif isinstance(other, FinishTimeConstant):
            return -1
        else:
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Invalid time information: "+self.label)
    def __repr__(self):
        return "NoTimeConstant("+self.label+")"


START_TIME = StartTimeConstant()
FINISH_TIME = FinishTimeConstant()

def timeslice(inslice):
    '''Preprocess a time slice, replacing the start with START_TIME and
    stop with FINISH_TIME.'''
    if not isinstance(inslice, slice):
        return inslice

    if inslice.start is None:
        start = START_TIME
    else:
        start = inslice.start

    if inslice.stop is None:
        stop = FINISH_TIME
    else:
        stop = inslice.stop

    if stop<=start:
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs(
            "Zero or negative length time slice.")

    return slice(start, stop, inslice.step)


class TimeTerm(object):
    '''A form evaluated at a point in time or over a time integral.'''
    def __init__(self, form, time):
        self.form = form
        self.time = time

    def __mul__(self, factor):
        return TimeTerm(factor * self.form, self.time)

    __rmul__ = __mul__

    def __div__(self, factor):
        return TimeTerm(1. / factor * self.form, self.time)

    def __repr__(self):
        return "TimeTerm("+self.form.__repr__()+",time = "+\
            repr(self.time)+")"

    def __neg__(self):
        return TimeTerm(-self.form,self.time)


class TimeForm(object):
    def __init__(self, terms):
        try:
            self.terms = list(terms)
        except TypeError:
            self.terms = [terms]

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        # Adding occurs by concatenating terms in the forms list.

        if isinstance(other, TimeForm):
            sum = TimeForm(self.terms + other.terms)
            return sum

        elif isinstance(other, ufl.form.Form):
            sum = TimeForm(self.terms + (other*dt[FINISH_TIME]).terms)
            return sum

        elif other in [0, 0.0]:
            return self

        else:
            return NotImplemented

    def __sub__(self, other):
        # Subtract by adding the negation of all the terms.

        if isinstance(other, TimeForm):
            sum = TimeForm(self.terms + [-term for term in other.terms])
            return sum

        else:
            return NotImplemented

    def __neg__(self):
        # Unary negation occurs by negating the terms.

        neg = TimeForm([-term for term in self.terms])
        return neg

    def __mul__(self, factor):
        return TimeForm([factor * term for term in self.terms])

    __rmul__ = __mul__

    def __div__(self, factor):
        return TimeForm([term / factor for term in self.terms])

    def __repr__(self):
        return "TimeForm("+repr(self.terms)+")"

    def is_functional(self):
        for term in self.terms:
            form = term.form

            if hasattr(form, 'compute_form_data'):
                fd = form.compute_form_data()
                if fd.rank != 0:
                    return False
            else:
                if term.form.arguments():
                    return False

            return True


class TimeMeasure(object):
    '''Define a measure for an integral over some interval in time.'''
    def __init__(self, interval = None):

        if interval is None:
            interval = slice(START_TIME,FINISH_TIME,None)

        self.interval = timeslice(interval)

    def __getitem__(object, key):

        return TimeMeasure(timeslice(key))

    def __rmul__(self, other):

        if isinstance(other, ufl.form.Form):
            # Multiplication with a form produces the TimeForm.
            return TimeForm(TimeTerm(other, self.interval))

        else:
            return NotImplemented

    def __repr__(self):
        return "TimeMeasure(interval = "+repr(self.interval)+")"

dt = TimeMeasure()

if __name__ == "__main__":
    from dolfin import *

    mesh = UnitSquare(2,2)

    U = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(U)

    F = v*dx

    TF = F*dt

    AT = at_time(F,0.0)
