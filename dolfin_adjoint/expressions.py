import backend
import collections
import copy

# Our equation may depend on Expressions, and those Expressions may have parameters
# (e.g. for time-dependent boundary conditions).
# In order to successfully replay the forward solve, we need to keep those parameters around.
# Here, we overload the Expression class to record all of the parameters

expression_attrs = collections.defaultdict(set)

if backend.__name__ == "dolfin":
    # A rant:
    # This had to be one of the most ridiculously difficult things in the whole
    # library, largely caused by the black magic associated with Expressions generally.
    # __new__, fecking metaclasses, the works --
    # I just want to subclass one of your classes, for heaven's sake!
    # (Subclassing Expression to do what I want is sublimely broken. Try it yourself
    # and go down the rabbit hole.)
    # Instead, I am forced into my own piece of underhanded trickery.

    expression_init = backend.Expression.__init__
    def __init__(self, *args, **kwargs):
        expression_init(self, *args, **kwargs)
        attr_list = expression_attrs[self]
        attr_list.union(list(kwargs.keys()))

    backend.Expression.__init__ = __init__

    expression_setattr = backend.Expression.__setattr__
    def __setattr__(self, k, v):
        expression_setattr(self, k, v)
        if k not in ["_ufl_element", "_ufl_shape", "_ufl_function_space", "_count", "_countedclass", "_repr", 
                     "_element", "this", "_value_shape", "user_parameters", "_hash"]: # <-- you may need to add more here as dolfin changes
            attr_list = expression_attrs[self]
            attr_list.add(k)
    backend.Expression.__setattr__ = __setattr__

def update_expressions(d):
    for expression in d:
        expr_dict = d[expression]
        for k in expr_dict:
            backend.Expression.__setattr__(expression, k, expr_dict[k])

def freeze_dict():
    new_dict = {}
    for expression in expression_attrs:
        attr_list = expression_attrs[expression]
        new_dict[expression] = {}

        for attr in attr_list:
            new_dict[expression][attr] = copy.copy(getattr(expression, attr))

    return new_dict
