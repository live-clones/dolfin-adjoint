import backend
import copy
from . import utils
from . import caching

backend_assemble_multimesh = backend.assemble_multimesh
def assemble_multimesh(*args, **kwargs):
    """When a multimesh form is assembled, the information about its nonlinear
    dependencies is lost, and it is no longer easy to manipulate. Therefore,
    dolfin_adjoint overloads the :py:func:`dolfin.assemble_multimesh` function
    to *attach the form to the assembled object*. This lets the automatic
    annotation work, even when the user calls the lower-level :py:data:`solve(A,
    x, b)`.
    """
    form = args[0]
    caching.assembled_fwd_forms.add(form)
    cache = kwargs.pop("cache", False)

    to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

    output = backend_assemble_multimesh(*args, **kwargs)
    if not isinstance(output, float) and to_annotate:
        output.form = form
        output.assemble_system = False

    if cache:
        caching.assembled_adj_forms[form] = output

    return output

# TODO: Overload MultiMeshPeriodicBC once it is implemented in DOLFIN

if hasattr(backend, 'MultiMeshDirichletBC'):
    multimesh_dirichlet_bc_apply = backend.MultiMeshDirichletBC.apply
    def adjoint_multimesh_dirichlet_bc_apply(self, *args, **kwargs):
        for arg in args:
            if not hasattr(arg, 'bcs'):
                arg.bcs = []
            arg.bcs.append(self)
        return multimesh_dirichlet_bc_apply(self, *args, **kwargs)
    backend.MultiMeshDirichletBC.apply = adjoint_multimesh_dirichlet_bc_apply

multimesh_function_vector = backend.MultiMeshFunction.vector
def adjoint_multimesh_function_vector(self):
    vec = multimesh_function_vector(self)
    vec.function = self
    return vec
backend.MultiMeshFunction.vector = adjoint_multimesh_function_vector

# TODO: Overload multimesh_assemble_system once it is implemented in DOLFIN
