from __future__ import print_function
import math
import numpy

import libadjoint
from backend import info_red, info_blue, info, warning
import backend
from . import adjglobals
from . import compatibility
from . import constant
from .enlisting import enlist
from .compatibility import gather  # NOQA
from .misc import noannotations


def scale(obj, factor):
    """ A generic function to scale Functions,
        Constants and lists, numpy arrays, ...
    """

    if hasattr(obj, "function_space"):
        # dolfin.Function of dolfin.MultiMeshFunctionSpace
        if isinstance(obj.function_space(), compatibility.multi_mesh_function_space_type):
            scaled_obj = backend.MultiMeshFunction(obj.function_space(), factor * obj.vector())
        else:
            scaled_obj = backend.Function(obj.function_space(), factor * obj.vector())
    elif isinstance(obj, backend.Constant):
        # dolfin.Constant
        scaled_obj = backend.Constant(factor * constant_to_array(obj))
    else:
        # Lists, numpy arrays, ...
        try:
            scaled_obj = factor * obj
        except TypeError:
            # Lists of dolfin objects?
            scaled_obj = [scale(o, factor) for o in obj]

    return scaled_obj

def homogenize(bc):
    bc_copy = compatibility.bc(bc)
    bc_copy.homogenize()
    return bc_copy

def constant_to_array(c):
    """ Converts a Constant to a numpy.array. """

    a = numpy.zeros(c.value_size())
    p = numpy.zeros(c.value_size())
    c.eval(a, p)
    if len(a) == 1:
        r = a[0]

    # Make sure input and output have the same shape
    assert c.ufl_shape == r.shape

    return r


def convergence_order(errors, base = 2):
    import math

    orders = [0.0] * (len(errors)-1)
    for i in range(len(errors)-1):
        try:
            orders[i] = math.log(errors[i]/errors[i+1], base)
        except ZeroDivisionError:
            orders[i] = numpy.nan

    return orders

def test_initial_condition_adjoint(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
    '''forward must be a function that takes in the initial condition (ic) as a backend.Function
       and returns the functional value by running the forward run:

         func = J(ic)

       final_adjoint is the adjoint associated with the initial condition
       (usually the last adjoint equation solved).

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the adjoint is working
       correctly.'''

    from . import function

    # We will compute the gradient of the functional with respect to the initial condition,
    # and check its correctness with the Taylor remainder convergence test.
    info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")

    # First run the problem unperturbed
    ic_copy = ic.copy(deepcopy=True)
    f_direct = J(ic_copy)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = function.Function(ic.function_space())
        compatibility.randomise(perturbation_direction)

    # Run the forward problem for various perturbed initial conditions
    functional_values = []
    perturbations = []
    perturbation_sizes = [seed/(2**i) for i in range(5)]
    for perturbation_size in perturbation_sizes:
        perturbation = perturbation_direction.copy(deepcopy=True)
        vec = perturbation.vector()
        vec *= perturbation_size
        perturbations.append(perturbation)

        perturbed_ic = ic.copy(deepcopy=True)
        vec = perturbed_ic.vector()
        vec += perturbation.vector()

        functional_values.append(J(perturbed_ic))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

    info("Taylor remainder without adjoint information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

    adjoint_vector = final_adjoint.vector()

    with_gradient = []
    gradient_fd   = []
    for i in range(len(perturbations)):
        gradient_fd.append((functional_values[i] - f_direct)/perturbation_sizes[i])

        remainder = abs(functional_values[i] - f_direct - adjoint_vector.inner(perturbations[i].vector()))
        with_gradient.append(remainder)

    info("Taylor remainder with adjoint information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

    info("Gradients (finite differencing): " + str(gradient_fd))
    info("Gradient (adjoint): " + str(adjoint_vector.inner(perturbation_direction.vector())))

    return min(convergence_order(with_gradient))

def tlm_dolfin(control, forget=False):
    for i in range(adjglobals.adjointer.equation_count):
        (tlm_var, output) = adjglobals.adjointer.get_tlm_solution(i, control)

        storage = libadjoint.MemoryStorage(output)
        storage.set_overwrite(True)
        adjglobals.adjointer.record_variable(tlm_var, storage)

        if forget is None:
            pass
        elif forget:
            adjglobals.adjointer.forget_tlm_equation(i)
        else:
            adjglobals.adjointer.forget_tlm_values(i)

    return output

def test_initial_condition_tlm(J, dJ, ic, seed=0.01, perturbation_direction=None):
    '''forward must be a function that takes in the initial condition (ic) as a backend.Function
       and returns the functional value by running the forward run:

         func = J(ic)

       final_adjoint is the tangent linear variable for the solution on which the functional depends
       (usually the last TLM equation solved).

       dJ must be the derivative of the functional with respect to its argument, evaluated and assembled at
       the unperturbed solution (a backend Vector).

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the TLM is working
       correctly.'''

    import controls

    # We will compute the gradient of the functional with respect to the initial condition,
    # and check its correctness with the Taylor remainder convergence test.
    info_blue("Running Taylor remainder convergence analysis for the tangent linear model... ")

    adj_var = adjglobals.adj_variables[ic]; adj_var.timestep = 0
    if not adjglobals.adjointer.variable_known(adj_var):
        info_red(str(adj_var) + " not known.")
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your initial condition must be the /exact same Function/ as the initial condition used in the forward model.")

    # First run the problem unperturbed
    ic_copy = backend.Function(ic)
    f_direct = J(ic_copy)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = backend.Function(ic.function_space())
        compatibility.randomise(perturbation_direction)

    # Run the forward problem for various perturbed initial conditions
    functional_values = []
    perturbations = []
    for perturbation_size in [seed/(2**i) for i in range(5)]:
        perturbation = backend.Function(perturbation_direction)
        vec = perturbation.vector()
        vec *= perturbation_size
        perturbations.append(perturbation)

        perturbed_ic = backend.Function(ic)
        vec = perturbed_ic.vector()
        vec += perturbation.vector()

        functional_values.append(J(perturbed_ic))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

    info("Taylor remainder without tangent linear information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without tangent linear information (should all be 1): " + str(convergence_order(no_gradient)))

    with_gradient = []
    for i in range(len(perturbations)):
        param = controls.FunctionControl(ic, perturbations[i])
        final_tlm = tlm_dolfin(param, forget=False).data
        remainder = abs(functional_values[i] - f_direct - final_tlm.vector().inner(dJ))
        with_gradient.append(remainder)

    info("Taylor remainder with tangent linear information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with tangent linear information (should all be 2): " + str(convergence_order(with_gradient)))

    return min(convergence_order(with_gradient))

def test_initial_condition_adjoint_cdiff(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
    '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
       and returns the functional value by running the forward run:

         func = J(ic)

       final_adjoint is the adjoint associated with the initial condition
       (usually the last adjoint equation solved).

       This function returns the order of convergence of the Taylor
       series remainder of central finite differencing, which should be 3
       if the adjoint is working correctly.'''

    # We will compute the gradient of the functional with respect to the initial condition,
    # and check its correctness with the Taylor remainder convergence test.
    info_blue("Running central differencing Taylor remainder convergence analysis for the adjoint model ... ")

    # First run the problem unperturbed
    ic_copy = backend.Function(ic)
    f_direct = J(ic_copy)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = backend.Function(ic.function_space())
        compatibility.randomise(perturbation_direction)

    # Run the forward problem for various perturbed initial conditions
    functional_values_plus = []
    functional_values_minus = []
    perturbations = []
    perturbation_sizes = [seed/(2**i) for i in range(4)]
    for perturbation_size in perturbation_sizes:
        perturbation = backend.Function(perturbation_direction)
        vec = perturbation.vector()
        vec *= perturbation_size
        perturbations.append(perturbation)

        perturbation = backend.Function(perturbation_direction)
        vec = perturbation.vector()
        vec *= perturbation_size/2.0

        perturbed_ic = backend.Function(ic)
        vec = perturbed_ic.vector()
        vec += perturbation.vector()
        functional_values_plus.append(J(perturbed_ic))

        perturbed_ic = backend.Function(ic)
        vec = perturbed_ic.vector()
        vec -= perturbation.vector()
        functional_values_minus.append(J(perturbed_ic))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(functional_values_plus[i] - functional_values_minus[i]) for i in range(len(functional_values_plus))]

    info("Taylor remainder without adjoint information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

    adjoint_vector = final_adjoint.vector()

    with_gradient = []
    gradient_fd   = []
    for i in range(len(perturbations)):
        gradient_fd.append((functional_values_plus[i] - functional_values_minus[i])/perturbation_sizes[i])

        remainder = abs(functional_values_plus[i] - functional_values_minus[i] - adjoint_vector.inner(perturbations[i].vector()))
        with_gradient.append(remainder)

    info("Taylor remainder with adjoint information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with adjoint information (should all be 3): " + str(convergence_order(with_gradient)))

    info("Gradients (finite differencing): " + str(gradient_fd))
    info("Gradient (adjoint): " + str(adjoint_vector.inner(perturbation_direction.vector())))

    return min(convergence_order(with_gradient))

def test_scalar_parameter_adjoint(J, a, dJda, seed=None):
    info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")

    functional_values = []
    f_direct = J(a)

    if seed is None:
        seed = float(a)/5.0
        if seed == 0.0:
            seed = 0.1

    perturbations = [seed / (2**i) for i in range(5)]

    for da in (backend.Constant(float(a) + x) for x in perturbations):
        functional_values.append(J(da))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

    info("Taylor remainder without adjoint information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

    with_gradient = []
    gradient_fd   = []
    for i in range(len(perturbations)):
        gradient_fd.append((functional_values[i] - f_direct)/perturbations[i])

        remainder = abs(functional_values[i] - f_direct - float(dJda)*perturbations[i])
        with_gradient.append(remainder)

    info("Taylor remainder with adjoint information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

    info("Gradients (finite differencing): " + str(gradient_fd))
    info("Gradient (adjoint): " + str(dJda))

    return min(convergence_order(with_gradient))

def test_scalar_parameters_adjoint(J, a, dJda, seed=0.1):
    info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")

    functional_values = []
    f_direct = J(a)

    a = numpy.array([float(x) for x in a])
    dJda = numpy.array(dJda)

    perturbation_direction = a/5.0
    perturbation_sizes = [seed / (2**i) for i in range(5)]
    perturbations = [a * i for i in perturbation_sizes]
    for x in perturbations:
        da = [backend.Constant(a[i] + x[i]) for i in range(len(a))]
        functional_values.append(J(da))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

    info("Taylor remainder without adjoint information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

    with_gradient = []
    for i in range(len(perturbations)):
        remainder = abs(functional_values[i] - f_direct - numpy.dot(dJda, perturbations[i]))
        with_gradient.append(remainder)

    info("Taylor remainder with adjoint information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

    return min(convergence_order(with_gradient))

def test_gradient_array(J, dJdx, x, seed = 0.01, perturbation_direction = None):
    '''Checks the correctness of the derivative dJ.
       x must be an array that specifies at which point in the parameter space
       the gradient is to be checked, and dJdx must be an array containing the gradient.
       The function J(x) must return the functional value.

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the gradient is correct.'''

    # We will compute the gradient of the functional with respect to the initial condition,
    # and check its correctness with the Taylor remainder convergence test.
    info("Running Taylor remainder convergence analysis to check the gradient ... ")

    # First run the problem unperturbed
    j_direct = J(x)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = x.copy()
        compatibility.randomise(perturbation_direction)

    # Run the forward problem for various perturbed initial conditions
    functional_values = []
    perturbations = []
    perturbation_sizes = [seed/(2**i) for i in range(5)]
    for perturbation_size in perturbation_sizes:
        perturbation = perturbation_direction.copy() * perturbation_size
        perturbations.append(perturbation)

        perturbed_x = x.copy() + perturbation
        functional_values.append(J(perturbed_x))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in functional_values]

    info("Absolute functional evaluation differences: %s" % str(no_gradient))
    info("Convergence orders for Taylor remainder without adjoint information (should all be 1): %s" % str(convergence_order(no_gradient)))

    with_gradient = []
    for i in range(len(perturbations)):
        remainder = abs(functional_values[i] - j_direct - numpy.dot(perturbations[i], dJdx))
        with_gradient.append(remainder)

    if min(with_gradient + no_gradient) < 1e-16:
        info("Warning: The Taylor remainders are close to machine precision (< %s). Try increasing the seed value in case the Taylor remainder test fails." % min(with_gradient + no_gradient))

    info("Absolute functional evaluation differences with adjoint: %s" % str(with_gradient))
    info("Convergence orders for Taylor remainder with adjoint information (should all be 2): %s" % str(convergence_order(with_gradient)))

    return min(convergence_order(with_gradient))

def taylor_remainder_with_gradient(m, Jm, dJdm, functional_value, perturbation, ic=None):
    """ Compute the Taylor remainder with the provided gradient information. Note that this
    computes the remainder for just one functional value. """
    from . import controls
    if isinstance(m, controls.ConstantControl):
        remainder = abs(functional_value - Jm - float(dJdm)*perturbation)
    elif isinstance(m, controls.ConstantControls):
        remainder = abs(functional_value - Jm - numpy.dot(dJdm, perturbation))
    elif isinstance(m, controls.FunctionControl):
        remainder = abs(functional_value - Jm - dJdm.vector().inner(perturbation.vector()))
    return remainder

@noannotations
def taylor_test(J, m, Jm, dJdm, HJm=None, seed=None, perturbation_direction=None, value=None):
    '''J must be a function that takes in a parameter value m and returns the value
       of the functional:

         func = J(m)

       Jm is the value of the function J at the parameter m.
       dJdm is the gradient of J evaluated at m, to be tested for correctness.

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the adjoint is working
       correctly.

       If HJm is not None, the Taylor test will also attempt to verify the
       correctness of the Hessian. HJm should be a callable which takes in a
       direction and returns the Hessian of the functional in that direction
       (i.e., takes in a vector and returns a vector). In that case, an additional
       Taylor remainder is computed, which should converge at order 3 if the Hessian
       is correct.'''

    from . import controls
    from .controls import ListControl

    info_blue("Running Taylor remainder convergence test ... ")

    if isinstance(m, list):
        m = ListControl(m)

    # Handle the multi-control case
    # We do this by performing a separate Taylor test for each control.
    if isinstance(m, controls.ListControl):
        return _taylor_test_multi_control(J, m, Jm, dJdm, HJm, seed, perturbation_direction, value)
    else:
        return _taylor_test_single_control(J, m, Jm, dJdm, HJm, seed, perturbation_direction, value)


def _taylor_test_multi_control(J, m, Jm, dJdm, HJm, seed, perturbation_direction, value):
    if perturbation_direction is None:
        perturbation_direction = [None] * len(m.controls)
    perturbation_direction = enlist(perturbation_direction)

    if value is None:
        value = [None] * len(m.controls)
    value = enlist(value)

    # Create a deep copy of the initial control values
    m_cpy = []
    for c in m:
        if isinstance(c.data(), backend.Function):
            m_cpy.append(c.data().copy(deepcopy=True))
        elif isinstance(c.data(), backend.MultiMeshFunction):
            m_cpy.append(c.data().copy(deepcopy=True))
        else:
            m_cpy.append(backend.Constant(c.data()))

    # Build a objective version restricted to the i'th control
    def J_cmp(J, i):
        m_values = list(m_cpy)
        def out(x):
            m_values[i] = x
            return J(m_values)
        return out

    # A Hessian version restricted to the i'th control
    if HJm is not None:
        HJm_cmp = lambda i: lambda x: HJm.__class__(HJm.J, m[i])(x)
    else:
        HJm_cmp = lambda i: None

    # Perform the Taylor tests for each control
    min_conv = 1e10
    for i in range(len(m.controls)):
        print("\nRunning Taylor test for control {}".format(i))
        conv = _taylor_test_single_control(J_cmp(J, i), m[i], Jm, dJdm[i],
                HJm_cmp(i), seed, perturbation_direction[i], value[i])
        min_conv = min(min_conv, conv)

    return min_conv


def _taylor_test_single_control(J, m, Jm, dJdm, HJm, seed, perturbation_direction, value):
    from . import function, controls

    # Check inputs
    if not isinstance(m, libadjoint.Parameter):
        raise ValueError("m must be a valid control instance.")

    def get_const(val):
        if isinstance(val, str):
            return float(constant.constant_values[val])
        else:
            return float(val)

    def get_value(param, value):
        if value is not None:
            return value
        else:
            try:
                return param.data()
            except libadjoint.exceptions.LibadjointErrorNeedValue:
                info_red("Do you need to pass forget=False to compute_gradient?")
                raise

    # First, compute perturbation sizes.
    seed_default = 0.01
    if seed is None:
        if isinstance(m, controls.ConstantControl):
            seed = get_const(m.a) / 5.0

            if seed == 0.0: seed = 0.1
        elif isinstance(m, controls.FunctionControl):
            ic = get_value(m, value)
            if len(ic.vector()) == 1: # our control is in R
                seed = float(ic) / 5.0
            else:
                seed = seed_default
        else:
            seed = seed_default

    perturbation_sizes = [seed/(2.0**i) for i in range(5)]

    # Next, compute the perturbation direction.
    if perturbation_direction is None:
        if isinstance(m, controls.ConstantControl):
            perturbation_direction = 1
        elif isinstance(m, controls.ConstantControls):
            perturbation_direction = numpy.array([get_const(x)/5.0 for x in m.v])
        elif isinstance(m, controls.FunctionControl):
            ic = get_value(m, value)

            # Check for MultiMeshFunction_space
            if isinstance(ic.function_space(), compatibility.multi_mesh_function_space_type):
                perturbation_direction = backend.MultiMeshFunction(ic.function_space())
            else:
                perturbation_direction = function.Function(ic.function_space())

            compatibility.randomise(perturbation_direction)

        else:
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to compute a perturbation direction")
    else:
        if isinstance(m, controls.FunctionControl):
            ic = get_value(m, value)
        elif isinstance(m, controls.ConstantControl):
            perturbation_direction = float(perturbation_direction)

    # So now compute the perturbations:
    if not isinstance(perturbation_direction, (backend.Function, backend.MultiMeshFunction)):
        perturbations = [x*perturbation_direction for x in perturbation_sizes]
    else:
        perturbations = []
        for x in perturbation_sizes:
            perturbation = perturbation_direction.copy(deepcopy=True)
            vec = perturbation.vector()
            vec *= x
            perturbations.append(perturbation)

    # And now the perturbed inputs:
    if isinstance(m, controls.ConstantControl):
        pinputs = [backend.Constant(get_const(m.a) + x) for x in perturbations]
    elif isinstance(m, controls.ConstantControls):
        a = numpy.array([get_const(x) for x in m.v])

        def make_const(arr):
            return [backend.Constant(x) for x in arr]

        pinputs = [make_const(a + x) for x in perturbations]
    elif isinstance(m, controls.FunctionControl):
        pinputs = []
        for x in perturbations:
            pinput = x.copy(deepcopy=True)
            vec = pinput.vector()
            vec += ic.vector()
            pinputs.append(pinput)

    # Issue 34: We must evaluate HJm before we evaluate the tape at the
    # perturbed controls below.
    if HJm is not None:
        HJm_values = []
        for perturbation in perturbations:
            HJmp = HJm(perturbation)
            HJm_values.append(HJmp)

    # At last: the common bit!
    functional_values = []
    for pinput in pinputs:
        Jp = J(pinput)
        functional_values.append(Jp)

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_J - Jm) for perturbed_J in functional_values]

    info("Taylor remainder without gradient information: " + str(no_gradient))
    info("Convergence orders for Taylor remainder without gradient information (should all be 1): " + str(convergence_order(no_gradient)))

    with_gradient = []
    for i in range(len(perturbations)):
        if isinstance(m, controls.ConstantControl) or isinstance(m, controls.ConstantControls):
            remainder = taylor_remainder_with_gradient(m, Jm, dJdm, functional_values[i], perturbations[i])
        else:
            remainder = taylor_remainder_with_gradient(m, Jm, dJdm, functional_values[i], perturbations[i], ic=ic)
        with_gradient.append(remainder)

    if min(with_gradient + no_gradient) < 1e-16:
        warning("Warning: The Taylor remainders are close to machine precision (< %s). Try increasing the seed value in case the Taylor remainder test fails." % min(with_gradient + no_gradient))

    info("Taylor remainder with gradient information: " + str(with_gradient))
    info("Convergence orders for Taylor remainder with gradient information (should all be 2): " + str(convergence_order(with_gradient)))

    if HJm is not None:
        with_hessian = []
        if isinstance(m, controls.ConstantControl):
            for i in range(len(perturbations)):
                remainder = abs(functional_values[i] - Jm - float(dJdm)*perturbations[i] - 0.5*perturbations[i]*HJm_values[i])
                with_hessian.append(remainder)
        elif isinstance(m, controls.ConstantControls):
            for i in range(len(perturbations)):
                remainder = abs(functional_values[i] - Jm - numpy.dot(dJdm, perturbations[i]) - 0.5*numpy.dot(perturbations[i], HJm_values[i]))
                with_hessian.append(remainder)
        elif isinstance(m, controls.FunctionControl):
            for i in range(len(perturbations)):
                remainder = abs(functional_values[i] - Jm - dJdm.vector().inner(perturbations[i].vector()) - 0.5*perturbations[i].vector().inner(HJm_values[i].vector()))
                with_hessian.append(remainder)

        info("Taylor remainder with Hessian information: " + str(with_hessian))
        info("Convergence orders for Taylor remainder with Hessian information (should all be 3): " + str(convergence_order(with_hessian)))
        return min(convergence_order(with_hessian))
    else:
        return min(convergence_order(with_gradient))


def taylor_test_expression(exp, V, seed=None):
    """
    Performs a Taylor test of an Expression with dependencies.

    exp: The expression to test
    V: A suitable function space on which the expression will be projected.
    seed: The initial perturbation coefficient for the taylor test.

    Warning: This function resets the adjoint tape! """
    from . import drivers, functional, projection
    from .controls import Control

    adjglobals.adj_reset()

    # Annotate test model
    s = projection.project(exp, V, annotate=True)
    mesh = V.mesh()

    Jform = s**2*backend.dx + exp*backend.dx(domain=mesh)

    J = functional.Functional(Jform)
    J0 = backend.assemble(Jform)

    controls = [Control(c) for c in exp.dependencies]
    dJd0 = drivers.compute_gradient(J, controls, forget=False)

    for i in range(len(controls)):
        def Jfunc(new_val):
            dep = exp.dependencies[i]

            # Remember the old dependency value for later
            old_val = float(dep)

            # Compute the functional value
            dep.assign(new_val)
            s = projection.project(exp, V, annotate=False)
            out = backend.assemble(s**2*backend.dx + exp*backend.dx(domain=mesh))

            # Restore the old dependency value
            dep.assign(old_val)

            return out

        #HJ = hessian(J, controls[i], warn=False)
        #minconv = taylor_test(Jfunc, controls[i], J0, dJd0[i], HJm=HJ)
        minconv = taylor_test(Jfunc, controls[i], J0, dJd0[i], seed=seed)

        if math.isnan(minconv):
            warning("Convergence order is not a number. Assuming that you \
have a linear or constant constraint dependency (e.g. check that the Taylor \
remainder are all 0).")
        else:
            if not minconv > 1.9:
                raise Exception("The Taylor test failed when checking the \
derivative with respect to the %i'th dependency." % (i+1))

    adjglobals.adj_reset()


def to_annotate(flag):
    '''Should dolfin-adjoint annotate this statement or not?'''
    if flag is None:
        return not backend.parameters["adjoint"]["stop_annotating"]

    if flag is True:
        if backend.parameters["adjoint"]["stop_annotating"]:
            raise AssertionError("The user insisted on annotation, but stop_annotating is True.")

    return flag

class DolfinAdjointVariable(libadjoint.Variable):
    ''' A wrapper class for Dolfin objects to store additional information such as
        a time step, a iteration counter and the type of the variable (adjoint, forward or tangent linear). '''

    def __init__(self, coefficient, timestep=None, iteration=None):
        ''' Create a DolfinAdjointVariable associated with the provided coefficient.

        If the coefficient is not known to dolfin_adjoint (i.e. if no equation for it was
        annotated), an Exception is thrown.

        By default, the DolfinAdjointVariable references the latest timestep and iteration number,
        but may be overwritten with the timestep and the iteration parameters. Negative values may
        be used to reference the backwards. '''

        super(DolfinAdjointVariable, self).__init__(var=adjglobals.adj_variables[coefficient].var)

        # First set the timestep, since the iteration_count call below depends on its value
        if timestep is not None:
            if timestep < 0:
                self.timestep = adjglobals.adjointer.timestep_count + timestep
            else:
                self.timestep = timestep

        if iteration is not None:
            if iteration < 0:
                self.iteration = self.iteration_count() + iteration
            else:
                self.iteration = iteration


    def tape_value(self, timestep=None, iteration=None):
        ''' Return the tape value associated with the variable (optionally for the given timestep and iteration). '''
        timestep = timestep or self.timestep
        iteration = iteration or self.iteration
        var = libadjoint.Variable(self.name, timestep, iteration, self.var.auxiliary)

        return adjglobals.adjointer.get_variable_value(var).data

    def iteration_count(self):
        ''' Return the annotated number of iterations at the variables timestep. '''
        return super(DolfinAdjointVariable, self).iteration_count(adjglobals.adjointer)

    def known_timesteps(self):
        ''' Return a list of timesteps for which this variable is annotated on the tape. '''
        ts = []
        var = libadjoint.Variable(self.name, 0, 0, self.var.auxiliary)
        for t in range(adjglobals.adjointer.timestep_count):
            var.timestep = t
            if adjglobals.adjointer.variable_known(var):
                ts.append(t)
        return ts

def get_identity_block(fn_space):
    assert fn_space is not None

    from . import adjlinalg
    block_name = "Identity: %s" % str(fn_space)
    if len(block_name) > int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]):
        block_name = block_name[0:int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"])-1]
    identity_block = libadjoint.Block(block_name)

    def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
        assert coefficient == 1
        if isinstance(fn_space, compatibility.function_space_type):
            return (adjlinalg.Matrix(adjlinalg.IdentityMatrix()), adjlinalg.Vector(backend.Function(fn_space)))
        else:
            return (adjlinalg.Matrix(adjlinalg.IdentityMatrix()), adjlinalg.Vector(backend.MultiMeshFunction(fn_space)))
    identity_block.assemble = identity_assembly_cb

    def identity_action_cb(variables, dependencies, hermitian, coefficient, input, context):
        output = input.duplicate()
        output.axpy(coefficient, input)
        return output

    identity_block.action = identity_action_cb

    return identity_block

def function_to_da_function(f):
    from . import function
    if isinstance(f, backend.MultiMeshFunction):
        return f
    if not isinstance(f, function.Function):
        # Wrap copy into a dolfin_adjoint.Function
        return function.Function(f.function_space(), f.vector())
    return f

def _has_multimesh(form):
    multimesh_integral_types = ["cutcell", "overlap", "interface"]

    for integral in form.integrals():
        if integral.integral_type() in multimesh_integral_types:
            return True

    for coefficient in form.coefficients():
        if hasattr(coefficient, "_V"):
            return True

    return False
