import ufl
import ufl.classes
import ufl.algorithms
import ufl.operators

import backend
import compatibility

import libadjoint
import libadjoint.exceptions

import hashlib
import copy
import os
import os.path
import random

import assembly
import expressions
import constant
import coeffstore
import adjrhs
import adjglobals
import adjlinalg
import misc
if backend.__name__ == "dolfin":
    import lusolver
import utils
import caching

def annotate(*args, **kwargs):
    '''This routine handles all of the annotation, recording the solves as they
    happen so that libadjoint can rewind them later.'''

    if 'matrix_class' in kwargs:
        matrix_class = kwargs['matrix_class']
        del kwargs['matrix_class']
    else:
        matrix_class = adjlinalg.Matrix

    if 'initial_guess' in kwargs:
        initial_guess = kwargs['initial_guess']
        del kwargs['initial_guess']
    else:
        initial_guess = False

    replace_map = False
    if 'replace_map' in kwargs:
        replace_map = kwargs['replace_map']
        del kwargs['replace_map']

    if isinstance(args[0], ufl.classes.Equation):
        # annotate !

        # Unpack the arguments, using the same routine as the real Dolfin solve call
        unpacked_args = compatibility._extract_args(*args, **kwargs)
        eq = unpacked_args[0]
        u  = unpacked_args[1]
        bcs = unpacked_args[2]
        J = unpacked_args[3]
        # create a deep copy of the parameters. They can be of type
        # backend.Parameters or just a list
        if type(unpacked_args[7]) == backend.Parameters:
            solver_parameters = backend.Parameters(unpacked_args[7])
        else:
            solver_parameters = copy.deepcopy(unpacked_args[7])

        if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):
            eq_lhs = eq.lhs
            eq_rhs = eq.rhs
            eq_bcs = bcs
            linear = True
        else:
            eq_lhs, eq_rhs = define_nonlinear_equation(eq.lhs, u)
            F = eq.lhs
            eq_bcs = []
            linear = False

    elif isinstance(args[0], compatibility.matrix_types()):
        linear = True
        try:
            eq_lhs = args[0].form
        except (KeyError, AttributeError) as e:
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs("dolfin_adjoint did not assemble your form, and so does not recognise your matrix. Did you from dolfin_adjoint import *?")

        try:
            eq_rhs = args[2].form
        except (KeyError, AttributeError) as e:
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs("dolfin_adjoint did not assemble your form, and so does not recognise your right-hand side. Did you from dolfin_adjoint import *?")

        u = args[1]
        u = u.function

        solver_parameters = {}

        try:
            solver_parameters["linear_solver"] = args[3]
        except IndexError:
            pass

        try:
            solver_parameters["preconditioner"] = args[4]
        except IndexError:
            pass

        try:
            eq_bcs = misc.uniq(args[0].bcs + args[2].bcs)
        except AttributeError:
            assert not hasattr(args[0], 'bcs') and not hasattr(args[2], 'bcs')
            eq_bcs = []
    else:
        print "args[0].__class__: ", args[0].__class__
        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to annotate your equation, sorry!")

    # Suppose we are solving for a variable w, and that variable shows up in the
    # coefficients of eq_lhs/eq_rhs.
    # Does that mean:
    #  a) the /previous value/ of that variable, and you want to timestep?
    #  b) the /value to be solved for/ in this solve?
    # i.e. is it timelevel n-1, or n?
    # if Dolfin is doing a linear solve, we want case a);
    # if Dolfin is doing a nonlinear solve, we want case b).
    # so if we are doing a nonlinear solve, we bump the timestep number here
    # /before/ we map the coefficients -> dependencies,
    # so that libadjoint records the dependencies with the right timestep number.
    if not linear:
        # Register the initial condition before the first nonlinear solve
        register_initial_conditions([[u, adjglobals.adj_variables[u]],], linear=False)
        var = adjglobals.adj_variables.next(u)
    else:
        var = None

    # Set up the data associated with the matrix on the left-hand side. This goes on the diagonal
    # of the 'large' system that incorporates all of the timelevels, which is why it is prefixed
    # with diag.
    diag_name = hashlib.md5(str(eq_lhs) + str(eq_rhs) + str(u) + str(random.random())).hexdigest() # we don't have a useful human-readable name, so take the md5sum of the string representation of the forms
    diag_deps = [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(eq_lhs) if hasattr(coeff, "function_space")]
    diag_coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(eq_lhs) if hasattr(coeff, "function_space")]

    if initial_guess and linear: # if the initial guess matters, we're going to have to add this in as a dependency of the system
        initial_guess_var = adjglobals.adj_variables[u]
        diag_deps.append(initial_guess_var)
        diag_coeffs.append(u)

    diag_block = libadjoint.Block(diag_name, dependencies=diag_deps, test_hermitian=backend.parameters["adjoint"]["test_hermitian"], test_derivative=backend.parameters["adjoint"]["test_derivative"])

    # Similarly, create the object associated with the right-hand side data.
    if linear:
        rhs = adjrhs.RHS(eq_rhs)
    else:
        rhs = adjrhs.NonlinearRHS(eq_rhs, F, u, bcs, mass=eq_lhs, solver_parameters=solver_parameters, J=J)


    # We need to check if this is the first equation,
    # so that we can register the appropriate initial conditions.
    # These equations are necessary so that libadjoint can assemble the
    # relevant adjoint equations for the adjoint variables associated with
    # the initial conditions.
    assert len(rhs.coefficients()) == len(rhs.dependencies())
    register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()) + zip(diag_coeffs, diag_deps), linear=linear, var=var)

    # c.f. the discussion above. In the linear case, we want to bump the
    # timestep number /after/ all of the dependencies' timesteps have been
    # computed for libadjoint.
    if linear:
        var = adjglobals.adj_variables.next(u)

    # With the initial conditions out of the way, let us now define the callbacks that
    # define the actions of the operator the user has passed in on the lhs of this equation.

    # Our equation may depend on Expressions, and those Expressions may have parameters
    # (e.g. for time-dependent boundary conditions).
    # In order to successfully replay the forward solve, we need to keep those parameters around.
    # In expressions.py, we overloaded the Expression class to record all of the parameters
    # as they are set. We're now going to copy that dictionary as it is at the annotation time,
    # so that we can get back to this exact state:
    frozen_expressions = expressions.freeze_dict()
    frozen_constants = constant.freeze_dict()

    def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
        '''This callback must conform to the libadjoint Python block assembly
        interface. It returns either the form or its transpose, depending on
        the value of the logical hermitian.'''

        assert coefficient == 1

        value_coeffs=[v.data for v in values]
        expressions.update_expressions(frozen_expressions)
        constant.update_constants(frozen_constants)
        eq_l = backend.replace(eq_lhs, dict(zip(diag_coeffs, value_coeffs)))

        kwargs = {"cache": eq_l in caching.assembled_fwd_forms} # should we cache our matrices on the way backwards?

        if hermitian:
            # Homogenise the adjoint boundary conditions. This creates the adjoint
            # solution associated with the lifted discrete system that is actually solved.
            adjoint_bcs = [utils.homogenize(bc) for bc in eq_bcs if isinstance(bc, backend.DirichletBC)] + [bc for bc in eq_bcs if not isinstance(bc, backend.DirichletBC)]
            if len(adjoint_bcs) == 0:
                adjoint_bcs = None
            else:
                adjoint_bcs = misc.uniq(adjoint_bcs)

            kwargs['bcs'] = adjoint_bcs
            kwargs['solver_parameters'] = solver_parameters
            kwargs['adjoint'] = True

            if initial_guess:
                kwargs['initial_guess'] = value_coeffs[dependencies.index(initial_guess_var)]

            if replace_map:
                kwargs['replace_map'] = dict(zip(diag_coeffs, value_coeffs))

            return (matrix_class(backend.adjoint(eq_l, reordered_arguments=ufl.algorithms.extract_arguments(eq_l)), **kwargs), adjlinalg.Vector(None, fn_space=u.function_space()))
        else:

            kwargs['bcs'] = misc.uniq(eq_bcs)
            kwargs['solver_parameters'] = solver_parameters
            kwargs['adjoint'] = False

            if initial_guess:
                kwargs['initial_guess'] = value_coeffs[dependencies.index(initial_guess_var)]

            if replace_map:
                kwargs['replace_map'] = dict(zip(diag_coeffs, value_coeffs))

            return (matrix_class(eq_l, **kwargs), adjlinalg.Vector(None, fn_space=u.function_space()))
    diag_block.assemble = diag_assembly_cb

    def diag_action_cb(dependencies, values, hermitian, coefficient, input, context):
        value_coeffs = [v.data for v in values]
        expressions.update_expressions(frozen_expressions)
        constant.update_constants(frozen_constants)
        eq_l = backend.replace(eq_lhs, dict(zip(diag_coeffs, value_coeffs)))

        if hermitian:
            eq_l = backend.adjoint(eq_l)

        output = coefficient * backend.action(eq_l, input.data)

        return adjlinalg.Vector(output)

    diag_block.action = diag_action_cb

    if len(diag_deps) > 0:
        # If this block is nonlinear (the entries of the matrix on the LHS
        # depend on any variable previously computed), then that will induce
        # derivative terms in the adjoint equations. Here, we define the
        # callback libadjoint will need to compute such terms.
        def derivative_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
            dolfin_variable = values[dependencies.index(variable)].data
            dolfin_values = [val.data for val in values]
            expressions.update_expressions(frozen_expressions)
            constant.update_constants(frozen_constants)

            current_form = backend.replace(eq_lhs, dict(zip(diag_coeffs, dolfin_values)))

            deriv = backend.derivative(current_form, dolfin_variable)
            args = ufl.algorithms.extract_arguments(deriv)
            deriv = backend.replace(deriv, {args[1]: contraction_vector.data}) # contract over the middle index

            # Assemble the G-matrix now, so that we can apply the Dirichlet BCs to it
            if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(coefficient*deriv))) == 0:
                return adjlinalg.Vector(None)

            G = coefficient * deriv

            if hermitian:
                output = backend.action(backend.adjoint(G), input.data)
            else:
                output = backend.action(G, input.data)

            return adjlinalg.Vector(output)
        diag_block.derivative_action = derivative_action

        def derivative_outer_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
            dolfin_variable = values[dependencies.index(variable)].data
            dolfin_values = [val.data for val in values]
            expressions.update_expressions(frozen_expressions)
            constant.update_constants(frozen_constants)

            current_form = backend.replace(eq_lhs, dict(zip(diag_coeffs, dolfin_values)))

            deriv = backend.derivative(current_form, dolfin_variable)
            args = ufl.algorithms.extract_arguments(deriv)
            deriv = backend.replace(deriv, {args[2]: contraction_vector.data}) # contract over the outer index

            # Assemble the G-matrix now, so that we can apply the Dirichlet BCs to it
            if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(coefficient*deriv))) == 0:
                return adjlinalg.Vector(None)

            G = coefficient * deriv

            if hermitian:
                output = backend.action(backend.adjoint(G), input.data)
            else:
                output = backend.action(G, input.data)

            return adjlinalg.Vector(output)
        diag_block.derivative_outer_action = derivative_outer_action

        def second_derivative_action(dependencies, values, inner_variable, inner_contraction_vector, outer_variable, outer_contraction_vector, hermitian, input, coefficient, context):
            dolfin_inner_variable = values[dependencies.index(inner_variable)].data
            dolfin_outer_variable = values[dependencies.index(outer_variable)].data
            dolfin_values = [val.data for val in values]
            expressions.update_expressions(frozen_expressions)
            constant.update_constants(frozen_constants)

            current_form = backend.replace(eq_lhs, dict(zip(diag_coeffs, dolfin_values)))

            deriv = backend.derivative(current_form, dolfin_inner_variable)
            args = ufl.algorithms.extract_arguments(deriv)
            deriv = backend.replace(deriv, {args[1]: inner_contraction_vector.data}) # contract over the middle index

            deriv = backend.derivative(deriv, dolfin_outer_variable)
            args = ufl.algorithms.extract_arguments(deriv)
            deriv = backend.replace(deriv, {args[1]: outer_contraction_vector.data}) # contract over the middle index

            # Assemble the G-matrix now, so that we can apply the Dirichlet BCs to it
            if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(coefficient*deriv))) == 0:
                return adjlinalg.Vector(None)

            G = coefficient * deriv

            if hermitian:
                output = backend.action(backend.adjoint(G), input.data)
            else:
                output = backend.action(G, input.data)

            return adjlinalg.Vector(output)
        diag_block.second_derivative_action = second_derivative_action

    eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)

    cs = adjglobals.adjointer.register_equation(eqn)
    do_checkpoint(cs, var, rhs)

    return linear

def solve(*args, **kwargs):
    '''This solve routine wraps the real Dolfin solve call. Its purpose is to annotate the model,
    recording what solves occur and what forms are involved, so that the adjoint and tangent linear models may be
    constructed automatically by libadjoint.

    To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).'''

    # First, decide if we should annotate or not.
    to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
    if to_annotate:
        linear = annotate(*args, **kwargs)

    # Avoid recursive annotation
    flag = misc.pause_annotation()
    try:
        ret = backend.solve(*args, **kwargs)
    except:
        raise
    finally:
        misc.continue_annotation(flag)

    if to_annotate:
        # Finally, if we want to record all of the solutions of the real forward model
        # (for comparison with a libadjoint replay later),
        # then we should record the value of the variable we just solved for.
        if backend.parameters["adjoint"]["record_all"]:
            if isinstance(args[0], ufl.classes.Equation):
                unpacked_args = compatibility._extract_args(*args, **kwargs)
                u  = unpacked_args[1]
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))
            elif isinstance(args[0], compatibility.matrix_types()):
                u = args[1].function
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))
            else:
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Don't know how to record, sorry")

    return ret

def define_nonlinear_equation(F, u):
    # Given an F := 0,
    # we write the equation for libadjoint's annotation purposes as
    # M.u = M.u + F(u)
    # as we need to have something on the diagonal in our big time system

    fn_space = u.function_space()
    test = backend.TestFunction(fn_space)
    trial = backend.TrialFunction(fn_space)

    mass = backend.inner(test, trial)*backend.dx

    return (mass, backend.action(mass, u) - F)

def adj_checkpointing(strategy, steps, snaps_on_disk,
        snaps_in_ram, verbose=False, replay = False, replay_comparison_tolerance = 1e-10):
    backend.parameters["adjoint"]["record_all"] = replay
    adjglobals.adjointer.set_checkpoint_strategy(strategy)
    adjglobals.adjointer.set_revolve_options(steps, snaps_on_disk, snaps_in_ram, verbose)
    adjglobals.adjointer.set_revolve_debug_options(replay, replay_comparison_tolerance)

def register_initial_conditions(coeffdeps, linear, var=None):
    for coeff, dep in coeffdeps:
        # If coeff is not known, it must be an initial condition.
        if not adjglobals.adjointer.variable_known(dep):

            if not linear: # don't register initial conditions for the first nonlinear solve.
                if dep == var:
                    continue

            if hasattr(coeff, "split_da"):
                if coeff.split is True:
                    errmsg = """
                    dolfin-adjoint does not support dolfin.Function.split().

                    To circumvent this, consider using split(func) rather
                    than func.split()
                    """
                    raise libadjoint.exceptions.LibadjointErrorNotImplemented(errmsg)

            register_initial_condition(coeff, dep)

def register_initial_condition(coeff, dep):
    fn_space = coeff.function_space()
    identity_block = utils.get_identity_block(fn_space)

    if backend.parameters["adjoint"]["record_all"]:
        adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(coeff)))

    init_rhs=adjlinalg.Vector(coeff).duplicate()
    init_rhs.axpy(1.0,adjlinalg.Vector(coeff))

    rhs = adjrhs.RHS(init_rhs)
    initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
    cs = adjglobals.adjointer.register_equation(initial_eq)
    assert adjglobals.adjointer.variable_known(dep)
    do_checkpoint(cs, dep, rhs)

def do_checkpoint(cs, var, rhs):
    if cs == int(libadjoint.constants.adj_constants["ADJ_CHECKPOINT_STORAGE_MEMORY"]):
        for coeff in adjglobals.adj_variables.keys():
            dep = adjglobals.adj_variables[coeff]
            # Do not checkpoint variables which are (yet) unknown to libadjoint
            if not adjglobals.adjointer.variable_known(dep):
                continue

            # Handle the Newton solve case:
            if dep == var:
                # We may need to checkpoint another variable if rhs is a NonlinearRHS and we need
                # to store the initial condition in order to replay the solve.
                if hasattr(rhs, 'ic_var') and rhs.ic_var is not None:
                    dep = rhs.ic_var
                else:
                    continue
                    for rdep in rhs.dependencies():
                        if rdep.name == dep.name:
                            dep = rdep
                            break

            adjglobals.mem_checkpoints.add(str(dep))
            adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(coeff), cs=True))

    elif cs == int(libadjoint.constants.adj_constants["ADJ_CHECKPOINT_STORAGE_DISK"]):

        for coeff in adjglobals.adj_variables.keys():
            dep = adjglobals.adj_variables[coeff]
            # Do not checkpoint variables which are (yet) unknown to libadjoint
            if not adjglobals.adjointer.variable_known(dep):
                continue

            if dep == var:
                # We may need to checkpoint another variable if rhs is a NonlinearRHS and we need
                # to store the initial condition in order to replay the solve.
                if hasattr(rhs, 'ic_var') and rhs.ic_var is not None:
                    dep = rhs.ic_var
                else:
                    continue

            adjglobals.disk_checkpoints.add(str(dep))
            adjglobals.adjointer.record_variable(dep, libadjoint.DiskStorage(adjlinalg.Vector(coeff), cs=True))


def record(val):
    adjglobals.adjointer.record_variable(adjglobals.adj_variables[val], libadjoint.MemoryStorage(adjlinalg.Vector(val)))
