import re
import ufl.algorithms
from ufl import Form
from backend import Constant

### A general dictionary that applies a key function before lookup
class KeyedDict(dict):
    def __init__(self, keyfunc, *args, **kwargs):
        self.keyfunc = keyfunc
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, x):
        return dict.__getitem__(self, self.keyfunc(x))

    def __setitem__(self, x, y):
        return dict.__setitem__(self, self.keyfunc(x), y)

    def __delitem__(self, x):
        return dict.__delitem__(self, self.keyfunc(x))

    def __contains__(self, x):
        return dict.__contains__(self, self.keyfunc(x))

    def clear(self):
        """ Delete all items.

            We need to be careful here and delete items in a specific order.
            This is crucial for MPI runs where destroying objects in different
            orders might result in MPI deadlocks.
        """
        def _comparable(key):
            # Form may be None when this is called during process cleanup
            if Form and isinstance(key, Form):
                return form.signature()
            return key
        try:
            keys = self.keys()
            for k in sorted(keys, key=_comparable):
                dict.__delitem__(self, k)
        except TypeError:
            # Something went wrong in the deallocation phase - try to exit
            # gracefully
            pass

    def __del__(self):
        self.clear()

### Stuff for LU caching

soa_to_adj = re.compile(r'\[(?P<func>Functional:.*?):.*\]')

# For caching strategies: a dictionary that maps adj_variable to LUSolver
# Not used by default

def lu_canonicalisation(var):
    # Return a string representation of var for indexing into the LU cache.

    s = str(var)

    # Since the SOA operator is always the same as the ADM, we can replace all
    # requests for SOA operators with ADM ones
    if var.type == 'ADJ_SOA':
        s = soa_to_adj.sub(r'[\1]', s).replace("SecondOrderAdjoint", "Adjoint")

    return s

lu_solvers = KeyedDict(keyfunc=lu_canonicalisation)

### Stuff for preassembly caching

def form_constants(form):
    constants = tuple([float(x) for x in ufl.algorithms.extract_coefficients(form) if isinstance(x, Constant)])
    return constants

def form_key(form):
    constants = form_constants(form)
    return (form, constants)

assembled_fwd_forms = set()
assembled_adj_forms = KeyedDict(keyfunc=form_key)

### Stuff for PointIntegralSolver caching
pis_fwd_to_tlm = {}
pis_fwd_to_adj = {}

# LocalSolver Cache
localsolvers = {}
