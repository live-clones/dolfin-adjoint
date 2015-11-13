"""
Handle en-listing and de-listing of controls, gradients, etc.

We want to support lists of controls [m1, m2, ...] to be optimised
simultaneously. But we also want the user to be able to say

dJ = compute_gradient(J, m)

rather than

dJ = compute_gradient(J, [m])[0]

.

Hence, we store everything internally in lists, but we need to know when
to de-list (i.e. when the user gave us a single control rather than
a list.) We do this by subclassing list and providing helper functions
to en-list and de-list appropriately.
"""

import collections

class Enlisted(list):
    pass

def enlist(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return Enlisted([x])

def delist(x, list_type):
    if isinstance(list_type, Enlisted):
        assert len(x) == 1
        return x[0]
    else:
        return x
