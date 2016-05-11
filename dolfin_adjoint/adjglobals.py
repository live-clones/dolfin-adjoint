import coeffstore
import expressions
import caching
import libadjoint
from petsc_krylov_solver import reset_petsc_krylov_solvers
from dolfin_adjoint import backend
if backend.__name__ == "dolfin":
    import lusolver

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()

mem_checkpoints = set()
disk_checkpoints = set()

adj_variables = coeffstore.CoeffStore()

def adj_start_timestep(time=0.0):
    '''Dolfin does not supply us with information about timesteps, and so more information
    is required from the user for certain features. This function should be called at the
    start of the time loop with the initial time (defaults to 0).

    See also: :py:func:`dolfin_adjoint.adj_inc_timestep`
    '''

    if not backend.parameters["adjoint"]["stop_annotating"]:
        adjointer.time.start(time)

def adj_inc_timestep(time=None, finished=False):
    '''Dolfin does not supply us with information about timesteps, and so more information
    is required from the user for certain features. This function should be called at
    the end of the time loop with two arguments:

      - :py:data:`time` -- the time at the end of the timestep just computed
      - :py:data:`finished` -- whether this is the final timestep.

    With this information, complex functional expressions using the :py:class:`Functional` class
    can be used.

    The finished argument is necessary because the final step of a functional integration must perform
    additional calculations.

    See also: :py:func:`dolfin_adjoint.adj_start_timestep`
    '''

    if not backend.parameters["adjoint"]["stop_annotating"]:
        adj_variables.increment_timestep()
        if time is not None:
            adjointer.time.next(time)

        if finished:
            adjointer.time.finish()

# A dictionary that saves the functionspaces of all checkpoint variables that have been saved to disk
checkpoint_fs = {}

function_names = set()

def adj_check_checkpoints():
    adjointer.check_checkpoints()

def adj_reset_cache():
    if backend.parameters["adjoint"]["debug_cache"]:
        backend.info_blue("Resetting solver cache")

    caching.assembled_fwd_forms.clear()
    caching.assembled_adj_forms.clear()
    caching.lu_solvers.clear()
    caching.localsolvers.clear()

    caching.pis_fwd_to_tlm.clear()
    caching.pis_fwd_to_adj.clear()

    if backend.__name__ == "dolfin":
        lusolver.lu_solvers = [None] * len(lusolver.lu_solvers)
        lusolver.adj_lu_solvers = [None] * len(lusolver.adj_lu_solvers)

    reset_petsc_krylov_solvers()

def adj_html(*args, **kwargs):
    '''This routine dumps the current state of the adjglobals.adjointer to a HTML visualisation.
    Use it like:

      - adj_html("forward.html", "forward") # for the equations recorded on the forward run
      - adj_html("adjoint.html", "adjoint") # for the equations to be assembled on the adjoint run
    '''
    return adjointer.to_html(*args, **kwargs)

def adj_reset():
    '''Forget all annotation, and reset the entire dolfin-adjoint state.'''
    adjointer.reset()
    expressions.expression_attrs.clear()
    adj_variables.__init__()
    function_names.__init__()
    adj_reset_cache()
    backend.parameters["adjoint"]["stop_annotating"] = False

# Map from FunctionSpace to LUSolver that has factorised the fsp mass matrix
fsp_lu = {}
