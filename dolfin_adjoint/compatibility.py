import backend
import numpy


if backend.__name__ == "firedrake":
    class Timer(object):
        def __init__(self, name):
            pass

        def start(self):
            pass

        def stop(self):
            return 0.0

        def value(self):
            return 0.0

    backend.Timer = Timer


def to_dict(d):
    if isinstance(d, dict):
        return d
    else:
        return d.to_dict()


def bc(bc):
    if backend.__name__ == "dolfin":
        return backend.DirichletBC(bc)
    else:
        return backend.DirichletBC(bc.function_space(), bc.function_arg, bc.sub_domain, method=bc.method)


def randomise(x):
    """ Randomises the content of x, where x can be a Function or a numpy.array.
    """

    if hasattr(x, "vector"):
        if backend.__name__ == "dolfin":
            vec = x.vector()
            vec_size = vec.local_size()
            vec.set_local(numpy.random.random(vec_size))
            vec.apply("")
        else:
            components = ("((float) rand()) / (float) RAND_MAX",)
            if isinstance(x, backend.Function):
                if(x.rank() > 0):
                    components *= len(x)
            temp = backend.Expression(components)
            x.interpolate(temp)
    else:
        # Make sure we get consistent values in MPI environments
        numpy.random.seed(seed=21)
        x[:] = numpy.random.random(len(x))


if hasattr(backend.Function, 'sub'):
    dolfin_sub    = backend.Function.sub
    def dolfin_adjoint_sub(self, idx, deepcopy=False):
        if backend.__name__ == "dolfin":
            out = dolfin_sub(self, idx, deepcopy=deepcopy)
        else:
            out = dolfin_sub(self, idx)
        out.super_idx = idx
        out.super_fn  = self
        return out


def assembled_rhs(b):
    if backend.__name__ == "dolfin":
        assembled_rhs = backend.Function(b.data).vector()
    else:
        assembled_rhs = backend.Function(b.data)
    return assembled_rhs


def assign_function_to_vector(x, b, function_space):
    """Assign the values of a backend.Function b to a adjlinalg.Vector x.

    If Firedrake is the backend, this currently creates a new Vector instead of modifying the one provided.
    """

    if backend.__name__ == "dolfin":
        x.data.vector()[:] = b.vector()
    else:
        from dolfin_adjoint.adjlinalg import Vector
        x = Vector(backend.Function(function_space).assign(b))
    return x

if backend.__name__ == "dolfin":
    solve = backend.fem.solving.solve
    matrix_types = lambda: (backend.cpp.Matrix, backend.GenericMatrix)
    _extract_args = backend.fem.solving._extract_args
    function_type = backend.cpp.Function
    function_space_type = backend.cpp.FunctionSpace

else:
    solve = backend.solving.solve
    matrix_types = lambda: backend.op2.base.Mat
    function_type = backend.Function
    function_space_type = (backend.functionspaceimpl.FunctionSpace,
                           backend.functionspaceimpl.WithGeometry,
                           backend.functionspaceimpl.MixedFunctionSpace)

    def _extract_args(*args, **kwargs):
        eq, u, bcs, _, _, _, _, solver_parameters, _, _, _ = backend.solving._extract_args(*args, **kwargs)
        return eq, u, bcs, None, None, None, None, solver_parameters
