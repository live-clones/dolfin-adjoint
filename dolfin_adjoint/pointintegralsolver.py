import dolfin
import ufl.algorithms
import libadjoint
from . import solving
from . import adjlinalg
from . import adjglobals
import hashlib
from . import utils
from . import caching
from . import expressions
from . import constant

class PointIntegralSolver(dolfin.PointIntegralSolver):
    def step(self, dt, annotate=None):

        timer = dolfin.Timer("Inner ODE step")
        to_annotate = utils.to_annotate(annotate)

        if to_annotate:
            scheme = self.scheme()
            var = scheme.solution()
            fn_space = var.function_space()

            current_var = adjglobals.adj_variables[var]
            if not adjglobals.adjointer.variable_known(current_var):
                solving.register_initial_conditions([(var, current_var)], linear=True)

            identity_block = utils.get_identity_block(fn_space)
            frozen_expressions = expressions.freeze_dict()
            frozen_constants = constant.freeze_dict()
            rhs = PointIntegralRHS(self, dt, current_var, frozen_expressions, frozen_constants)
            next_var = adjglobals.adj_variables.next(var)

            eqn = libadjoint.Equation(next_var, blocks=[identity_block], targets=[next_var], rhs=rhs)
            cs  = adjglobals.adjointer.register_equation(eqn)

        super(PointIntegralSolver, self).step(dt)

        if to_annotate:
            curtime = float(scheme.t())
            scheme.t().assign(curtime) # so that d-a sees the time update, which is implict in step

            solving.do_checkpoint(cs, next_var, rhs)

            if dolfin.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(next_var, libadjoint.MemoryStorage(adjlinalg.Vector(var)))
        timer.stop()

class PointIntegralRHS(libadjoint.RHS):
    def __init__(self, solver, dt, ic_var, frozen_expressions, frozen_constants):
        timer = dolfin.Timer("ODE adjoint init")
        self.solver = solver
        if hasattr(self.solver, 'reset'): self.solver.reset()
        self.dt = dt

        self.scheme = scheme = solver.scheme()
        self.time = float(scheme.t())
        self.form = scheme.rhs_form()
        self.fn_space = scheme.solution().function_space()

        self.coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]
        self.deps   = [adjglobals.adj_variables[coeff] for coeff in self.coeffs]

        solving.register_initial_conditions(zip(self.coeffs, self.deps), linear=True)

        self.ic_var = ic_var
        if scheme.solution() not in self.coeffs:
            self.coeffs.append(scheme.solution())
            self.deps.append(ic_var)

        self.frozen_expressions = frozen_expressions
        self.frozen_constants = frozen_constants
        timer.stop()

    def dependencies(self):
        return self.deps

    def coefficients(self):
        return self.coeffs

    def __str__(self):
        return hashlib.md5(str(self.form).encode('utf8')).hexdigest()

    def __call__(self, dependencies, values):

        timer = dolfin.Timer("ODE replay")
        expressions.update_expressions(self.frozen_expressions)
        constant.update_constants(self.frozen_constants)

        coeffs = [x for x in ufl.algorithms.extract_coefficients(self.scheme.rhs_form()) if hasattr(x, 'function_space')]
        for (coeff, value) in zip(coeffs, values):
            coeff.assign(value.data)

        self.scheme.t().assign(self.time)
        self.solver.step(self.dt)

        # FIXME: This form should actually be from before the solve.
        out = adjlinalg.Vector(self.scheme.solution())
        out.nonlinear_form = -self.dt*self.scheme.rhs_form()
        out.nonlinear_u = self.scheme.solution()
        out.nonlinear_bcs = []
        out.nonlinear_J = ufl.derivative(out.nonlinear_form, out.nonlinear_u)
        timer.stop()

        return out

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        expressions.update_expressions(self.frozen_expressions)
        constant.update_constants(self.frozen_constants)

        if not hermitian:
            timer = dolfin.Timer("ODE tlm step")
            if self.solver not in caching.pis_fwd_to_tlm:
                dolfin.info_blue("No TLM solver, creating ... ")
                creation_timer = dolfin.Timer("to_adm")
                if contraction_vector.data is not None:
                    tlm_scheme = self.scheme.to_tlm(contraction_vector.data)
                else:
                    tlm_scheme = self.scheme.to_tlm(dolfin.Function(self.fn_space))

                creation_time = creation_timer.stop()
                dolfin.info_red("TLM creation time: %s" % creation_time)

                tlm_solver = dolfin.PointIntegralSolver(tlm_scheme)
                tlm_solver.parameters.update(self.solver.parameters)
                caching.pis_fwd_to_tlm[self.solver] = tlm_solver
            else:
                tlm_solver = caching.pis_fwd_to_tlm[self.solver]
                tlm_scheme = tlm_solver.scheme()
                if contraction_vector.data is not None:
                    tlm_scheme.contraction.assign(contraction_vector.data)
                else:
                    tlm_scheme.contraction.vector().zero()

            coeffs = [x for x in ufl.algorithms.extract_coefficients(tlm_scheme.rhs_form()) if hasattr(x, 'function_space')]
            for (coeff, value) in zip(coeffs, values):
                coeff.assign(value.data)
            tlm_scheme.t().assign(self.time)

            tlm_solver.step(self.dt)

            timer.stop()
            return adjlinalg.Vector(tlm_scheme.solution())

        else:
            print("ODE adjoint step solving for {}".format(self.ic_var))
            timer = dolfin.Timer("ODE adjoint step")
            timer_inner = dolfin.Timer("ODE adjoint inner preparation")
            if self.solver not in caching.pis_fwd_to_adj:
                dolfin.info_blue("No ADM solver, creating ... ")
                creation_timer = dolfin.Timer("to_adm")
                if contraction_vector.data is not None:
                    adm_scheme = self.scheme.to_adm(contraction_vector.data)
                else:
                    adm_scheme = self.scheme.to_adm(dolfin.Function(self.fn_space))
                creation_time = creation_timer.stop()
                dolfin.info_red("ADM creation time: %s" % creation_time)

                adm_solver = dolfin.PointIntegralSolver(adm_scheme)
                adm_solver.parameters.update(self.solver.parameters)
                caching.pis_fwd_to_adj[self.solver] = adm_solver
            else:
                adm_solver = caching.pis_fwd_to_adj[self.solver]
                adm_scheme = adm_solver.scheme()
                if contraction_vector.data is not None:
                    adm_scheme.contraction.assign(contraction_vector.data)
                else:
                    adm_scheme.contraction.vector().zero()

            coeffs = [x for x in ufl.algorithms.extract_coefficients(adm_scheme.rhs_form()) if hasattr(x, 'function_space')]
            for (coeff, value) in zip(coeffs, values):
                coeff.assign(value.data)
            adm_scheme.t().assign(self.time)
            timer_inner.stop()

            timer_inner = dolfin.Timer("ODE adjoint inner step")
            adm_solver.step(self.dt)
            timer_inner.stop()

            timer.stop()
            return adjlinalg.Vector(adm_scheme.solution())

__all__ = ['PointIntegralSolver']
