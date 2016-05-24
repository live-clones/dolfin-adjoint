import re
import ufl.algorithms
from backend import Constant
from collections import OrderedDict


### A general dictionary that applies a key function before lookup
class KeyedDict(OrderedDict):
    def __init__(self, keyfunc, *args, **kwargs):
        self.keyfunc = keyfunc
        super(KeyedDict, self).__init__(*args, **kwargs)

    def __getitem__(self, x):
        return super(KeyedDict, self).__getitem__(self.keyfunc(x))

    def __setitem__(self, x, y):
        return super(KeyedDict, self).__setitem__(self.keyfunc(x), y)

    def __delitem__(self, x):
        return super(KeyedDict, self).__delitem__(self.keyfunc(x))

    def __contains__(self, x):
        return super(KeyedDict, self).__contains__(self.keyfunc(x))


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
    if isinstance(form, tuple):
        return form
    constants = form_constants(form)
    return (form, constants)

assembled_fwd_forms = set()
assembled_adj_forms = KeyedDict(keyfunc=form_key)

### Stuff for PointIntegralSolver caching
pis_fwd_to_tlm = {}
pis_fwd_to_adj = {}

# LocalSolver Cache
localsolvers = {}
