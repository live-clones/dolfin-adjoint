import backend
from assembly import assemble, assemble_system
# Functional
from functional import Functional

# Controls
from controls import Control
from controls import FunctionControl
from controls import ConstantControl
from controls import ConstantControls

from solving import solve, adj_checkpointing, annotate, record
from adjglobals import adj_start_timestep, adj_inc_timestep, adjointer, adj_check_checkpoints, adj_html, adj_reset
from gst import compute_gst, compute_propagator_matrix, perturbed_replay
from utils import convergence_order, DolfinAdjointVariable
from utils import taylor_test
from utils import taylor_test_expression
from drivers import replay_dolfin, compute_adjoint, compute_tlm, compute_gradient, hessian, compute_gradient_tlm
from misc import annotations

from variational_solver import NonlinearVariationalSolver, NonlinearVariationalProblem, LinearVariationalSolver, LinearVariationalProblem
from projection import project
from function import Function
from interpolation import interpolate
from constant import Constant
from timeforms import dt, TimeMeasure, START_TIME, FINISH_TIME

# Expose PDE-constrained optimization utilities
from optimization.optimization_problem import *
from optimization.optimization_solver import *
from optimization.moola_problem import *
from optimization.ipopt_solver import *
from optimization.optizelle_solver import *
from optimization.riesz_maps import *

from reduced_functional import ReducedFunctional
from reduced_functional_numpy import ReducedFunctionalNumPy, ReducedFunctionalNumpy
from optimization.constraints import InequalityConstraint, EqualityConstraint
from optimization.optimization import minimize, maximize, print_optimization_methods, minimise, maximise
from optimization.tao_solver import TAOSolver
if backend.__name__ == "dolfin":
    from multimeshfunction import MultiMeshFunction
    from multimesh_assembly import assemble_multimesh
    from newton_solver import NewtonSolver
    from krylov_solver import KrylovSolver
    from petsc_krylov_solver import PETScKrylovSolver
    from linear_solver import LinearSolver
    from lusolver import LUSolver
    from localsolver import LocalSolver
    from optimization.multistage_optimization import minimize_multistage
    from pointintegralsolver import *
    if hasattr(backend, 'FunctionAssigner'):
        from functionassigner import FunctionAssigner
