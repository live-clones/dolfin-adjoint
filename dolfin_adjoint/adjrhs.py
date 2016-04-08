import libadjoint
import backend
import ufl
import ufl.algorithms
import adjglobals
import adjlinalg
import utils

def find_previous_variable(var):
    ''' Returns the previous instance of the given variable. '''

    for timestep in range(var.timestep, -1, -1):
        prev_var = libadjoint.Variable(var.name, timestep, 0)

        if adjglobals.adjointer.variable_known(prev_var):
            prev_var.var.iteration = prev_var.iteration_count(adjglobals.adjointer) - 1
            return prev_var

    raise libadjoint.exceptions.LibadjointErrorInvalidInputs, 'No previous variable found'

def _extract_function_coeffs(form):
    for c in ufl.algorithms.extract_coefficients(form):
        if isinstance(c, backend.Function):
            yield c

class RHS(libadjoint.RHS):
    '''This class implements the libadjoint.RHS abstract base class for the Dolfin adjoint.
    It takes in a form, and implements the necessary routines such as calling the right-hand side
    and taking its derivative.'''
    def __init__(self, form):

        self.form=form

        if isinstance(self.form, ufl.form.Form):
            self.deps = [adjglobals.adj_variables[coeff] for coeff in _extract_function_coeffs(self.form)]
        else:
            self.deps = []

        if isinstance(self.form, ufl.form.Form):
            self.coeffs = [coeff for coeff in _extract_function_coeffs(self.form)]
        else:
            self.coeffs = []

    def __call__(self, dependencies, values):

        if isinstance(self.form, ufl.form.Form):

            dolfin_dependencies=[dep for dep in _extract_function_coeffs(self.form)]

            dolfin_values=[val.data for val in values]

            return adjlinalg.Vector(backend.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

        else:
            # RHS is a adjlinalg.Vector.
            assert isinstance(self.form, adjlinalg.Vector)
            return self.form


    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

        if contraction_vector.data is None:
            return adjlinalg.Vector(None)

        if isinstance(self.form, ufl.form.Form):
            # Find the dolfin Function corresponding to variable.
            dolfin_variable = values[dependencies.index(variable)].data

            dolfin_dependencies = [dep for dep in _extract_function_coeffs(self.form)]

            dolfin_values = [val.data for val in values]

            current_form = backend.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
            trial = backend.TrialFunction(dolfin_variable.function_space())

            d_rhs = backend.derivative(current_form, dolfin_variable, trial)

            if hermitian:
                action = backend.action(backend.adjoint(d_rhs), contraction_vector.data)
            else:
                action = backend.action(d_rhs, contraction_vector.data)

            return adjlinalg.Vector(action)
        else:
            # RHS is a adjlinalg.Vector. Its derivative is therefore zero.
            return adjlinalg.Vector(None)

    def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action_vector):

        if isinstance(self.form, ufl.form.Form):
            # Find the dolfin Function corresponding to variable.
            dolfin_inner_variable = values[dependencies.index(inner_variable)].data
            dolfin_outer_variable = values[dependencies.index(outer_variable)].data

            dolfin_dependencies = [dep for dep in _extract_function_coeffs(self.form)]

            dolfin_values = [val.data for val in values]

            current_form = backend.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
            trial = backend.TrialFunction(dolfin_outer_variable.function_space())

            d_rhs = backend.derivative(current_form, dolfin_inner_variable, inner_contraction_vector.data)
            d_rhs = ufl.algorithms.expand_derivatives(d_rhs)
            if len(d_rhs.integrals()) == 0:
                return None

            d_rhs = backend.derivative(d_rhs, dolfin_outer_variable, trial)
            d_rhs = ufl.algorithms.expand_derivatives(d_rhs)

            if len(d_rhs.integrals()) == 0:
                return None

            if hermitian:
                action = backend.action(backend.adjoint(d_rhs), action_vector.data)
            else:
                action = backend.action(d_rhs, action_vector.data)

            return adjlinalg.Vector(action)
        else:
            # RHS is a adjlinalg.Vector. Its derivative is therefore zero.
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("No derivative method for constant RHS.")

    def dependencies(self):

        return self.deps

    def coefficients(self):

        return self.coeffs

    def __str__(self):

        return hashlib.md5(str(self.form)).hexdigest()

class NonlinearRHS(RHS):
    '''For nonlinear problems, the source term isn't assembled in the usual way.
    If the nonlinear problem is given as
    F(u) = 0,
    we annotate it as
    M.u = M.u - F(u) .
    So in order to actually assemble the right-hand side term,
    we first need to solve F(u) = 0 to find the specific u,
    and then multiply that by the mass matrix.'''
    def __init__(self, form, F, u, bcs, mass, solver_parameters, J):
        '''form is M.u - F(u). F is the nonlinear equation, F(u) := 0.'''
        RHS.__init__(self, form)
        self.F = F
        self.u = u
        self.bcs = bcs
        self.mass = mass
        self.solver_parameters = solver_parameters
        self.J = J or backend.derivative(F, u)

        # We want to mark that the RHS term /also/ depends on
        # the previous value of u, as that's what we need to initialise
        # the nonlinear solver.
        var = adjglobals.adj_variables[self.u]
        self.ic_var = None

        if backend.parameters["adjoint"]["fussy_replay"]:
            can_depend = True
            try:
                prev_var = find_previous_variable(var)
            except:
                can_depend = False

            if can_depend:
                self.ic_var = prev_var
                self.deps += [self.ic_var]
                self.coeffs += [u]

            else:
                self.ic_copy = backend.Function(u)
                self.ic_var = None

    def __call__(self, dependencies, values):
        assert isinstance(self.form, ufl.form.Form)

        ic = self.u.function_space() # by default, initialise with a blank function in the solution FunctionSpace

        if hasattr(self, "ic_copy"):
            ic = self.ic_copy

        replace_map = {}

        for i in range(len(self.deps)):
            if self.deps[i] in dependencies:
                j = dependencies.index(self.deps[i])
                if self.deps[i] == self.ic_var:
                    ic = values[j].data # ahah, we have found an initial condition!
                else:
                    replace_map[self.coeffs[i]] = values[j].data

        current_F    = backend.replace(self.F, replace_map)
        current_J    = backend.replace(self.J, replace_map)
        u = backend.Function(ic)
        current_F    = backend.replace(current_F, {self.u: u})
        current_J    = backend.replace(current_J, {self.u: u})

        vec = adjlinalg.Vector(None)
        vec.nonlinear_form = current_F
        vec.nonlinear_u = u
        vec.nonlinear_bcs = self.bcs
        vec.nonlinear_J = current_J

        return vec

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        '''If variable is the variable for the initial condition, we want to ignore it,
        and set the derivative to zero. Assuming the solver converges, the sensitivity of
        the solution to the initial condition should be extremely small, and computing it
        is very difficult (one would have to do a little adjoint solve to compute it).
        Even I'm not that fussy.'''

        if variable == self.ic_var:
            deriv_value = values[dependencies.index(variable)].data
            return adjlinalg.Vector(None, fn_space=deriv_value.function_space())
        else:
            return RHS.derivative_action(self, dependencies, values, variable, contraction_vector, hermitian)

    def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action):
        '''If variable is the variable for the initial condition, we want to ignore it,
        and set the derivative to zero. Assuming the solver converges, the sensitivity of
        the solution to the initial condition should be extremely small, and computing it
        is very difficult (one would have to do a little adjoint solve to compute it).
        Even I'm not that fussy.'''

        if inner_variable == self.ic_var or outer_variable == self.ic_var:
            deriv_value = values[dependencies.index(outer_variable)].data
            return adjlinalg.Vector(None, fn_space=deriv_value.function_space())
        else:
            return RHS.second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action)

    def derivative_assembly(self, dependencies, values, variable, hermitian):
        replace_map = {}

        for i in range(len(self.deps)):
            if self.deps[i] == self.ic_var: continue
            j = dependencies.index(self.deps[i])
            replace_map[self.coeffs[i]] = values[j].data

        diff_var = values[dependencies.index(variable)].data

        current_form = backend.replace(self.form, replace_map)
        deriv = backend.derivative(current_form, diff_var)

        if hermitian:
            deriv = backend.adjoint(deriv)
            bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, backend.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, backend.DirichletBC)]
        else:
            bcs = self.bcs

        return adjlinalg.Matrix(deriv, bcs=bcs)

def adj_get_forward_equation(i):
    (fwd_var, lhs, rhs) = adjglobals.adjointer.get_forward_equation(i)

    # We needed to cheat the annotation when we registered a nonlinear solve.
    # However, if we want to actually differentiate the form (e.g. to compute
    # the dependency of the form on a ScalarParameter) we're going to need
    # the real F(u) = 0 back again. So let's fetch it here:
    if hasattr(rhs, 'nonlinear_form'):
        lhs = rhs.nonlinear_form
        fwd_var.nonlinear_u = rhs.nonlinear_u
        rhs = 0
    else:
        lhs = lhs.data
        rhs = rhs.data

    return (fwd_var, lhs, rhs)
