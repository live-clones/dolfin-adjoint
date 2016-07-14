import solving
import backend
if backend.__name__ == "dolfin":
    import backend.fem.projection
import misc
import libadjoint
import adjglobals
import adjlinalg
import utils

def project_dolfin(v, V=None, bcs=None, mesh=None, solver_type="cg", preconditioner_type="default", form_compiler_parameters=None, annotate=None, name=None):
    '''The project call performs an equation solve, and so it too must be annotated so that the
    adjoint and tangent linear models may be constructed automatically by libadjoint.

    To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
    cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
    computation (such as projecting fields to other function spaces for the purposes of
    visualisation).'''

    to_annotate = utils.to_annotate(annotate)

    if isinstance(v, backend.Expression) and (annotate is not True):
        to_annotate = False

    if isinstance(v, backend.Constant) and (annotate is not True):
        to_annotate = False

    out = backend.project(v=v, V=V, bcs=bcs, mesh=mesh, solver_type=solver_type, preconditioner_type=preconditioner_type, form_compiler_parameters=form_compiler_parameters)
    out = utils.function_to_da_function(out)

    if name is not None:
        out.adj_name = name
        out.rename(name, "a Function from dolfin-adjoint")

    if to_annotate:
        # reproduce the logic from project. This probably isn't future-safe, but anyway

        if V is None:
            V = backend.fem.projection._extract_function_space(v, mesh)
        if mesh is None:
            mesh = V.mesh()

        # Define variational problem for projection
        w = backend.TestFunction(V)
        Pv = backend.TrialFunction(V)
        a = backend.inner(w, Pv)*backend.dx(domain=mesh)
        L = backend.inner(w, v)*backend.dx(domain=mesh)

        solving.annotate(a == L, out, bcs, solver_parameters={"linear_solver": solver_type, "preconditioner": preconditioner_type, "symmetric": True})

        if backend.parameters["adjoint"]["record_all"]:
            adjglobals.adjointer.record_variable(adjglobals.adj_variables[out], libadjoint.MemoryStorage(adjlinalg.Vector(out)))

    return out

# In Firedrake, project wraps an actual variational solve, so there is
# no need for dolfin-adjoint to treat it specially. It is sufficient
# that the inner solve is annotated.
def project_firedrake(*args, **kwargs):

    annotate = kwargs.pop("annotate", None)

    to_annotate = utils.to_annotate(annotate)

    if isinstance(args[0], backend.Expression) and (annotate is not True):
        to_annotate = False

    if isinstance(args[0], backend.Constant) and (annotate is not True):
        to_annotate = False

    if to_annotate:
        result = backend.project(*args, **kwargs)
    else:
        flag = misc.pause_annotation()
        result = backend.project(*args, **kwargs)
        misc.continue_annotation(flag)

    return utils.function_to_da_function(result)


if backend.__name__ == "dolfin":
    project = project_dolfin
else:
    project = project_firedrake
