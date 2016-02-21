import backend
import copy

constant_values = {}
constant_objects = {}
scalar_parameters = []

class Constant(backend.Constant):
    '''The Constant class is overloaded so that you can give :py:class:`Constants` *names*. For example,

      .. code-block:: python

        nu = Constant(1.0e-4, name="Diffusivity")

      This allows you to refer to the :py:class:`Constant` by name throughout dolfin-adjoint, rather than
      needing to have the specific :py:class:`Constant` instance available.

      For more details, see :doc:`the dolfin-adjoint documentation </documentation/misc>`.'''

    def __init__(self, value, cell=None, name=None):
        backend.Constant.__init__(self, value, cell)
        if name is None:
            name = hash(self)

        self.adj_name = name

        if name in constant_values:
            backend.info_red("Warning: redefing constant with name %s" % name)

        constant_values[name] = value
        constant_objects[name] = self

    def assign(self, value):
        backend.Constant.assign(self, value)
        constant_values[self.adj_name] = value

def get_constant(a):
    if isinstance(a, Constant):
        return a
    else:
        return constant_objects[a]

def freeze_dict():
    new_dict = {}
    for name in constant_objects:
        new_dict[constant_objects[name]] = copy.copy(constant_values[name])

    return new_dict

def update_constants(d):
    for constant in d:
        name = constant.adj_name
        if name not in scalar_parameters:
            backend.Constant.assign(constant_objects[name], backend.Constant(d[constant]))
