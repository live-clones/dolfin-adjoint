import libadjoint
from controls import *
from backend import info_red, info_blue, info, info_green, parameters
import backend
import adjglobals
import backend
import constant
import adjresidual
import ufl.algorithms
from enlisting import enlist, delist
from numpy import ndarray
import misc

def replay_dolfin(forget=False, tol=0.0, stop=False):

    with misc.annotations(False):
        if not backend.parameters["adjoint"]["record_all"]:
            info_red("Warning: your replay test will be much more effective with dolfin.parameters['adjoint']['record_all'] = True.")

        success = True
        for i in range(adjglobals.adjointer.equation_count):
            (fwd_var, output) = adjglobals.adjointer.get_forward_solution(i)
            storage = libadjoint.MemoryStorage(output)
            storage.set_compare(tol=tol)
            storage.set_overwrite(True)
            out = adjglobals.adjointer.record_variable(fwd_var, storage)
            success = success and out

            if forget:
                adjglobals.adjointer.forget_forward_equation(i)

            if not out and stop:
                break

    return success

def compute_adjoint(functional, forget=True, ignore=[]):

    ignorelist = []
    for fn in ignore:
        if isinstance(fn, backend.Function):
            ignorelist.append(adjglobals.adj_variables[fn])
        elif isinstance(fn, str):
            ignorelist.append(libadjoint.Variable(fn, 0, 0))
        else:
            ignorelist.append(fn)

    for i in range(adjglobals.adjointer.timestep_count):
        adjglobals.adjointer.set_functional_dependencies(functional, i)

    for i in range(adjglobals.adjointer.equation_count)[::-1]:
        fwd_var = adjglobals.adjointer.get_forward_variable(i)
        if fwd_var in ignorelist:
            info("Ignoring the adjoint equation for %s" % fwd_var)
            continue

        (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, functional)
        if output.data:
            if backend.__name__ == "dolfin":
                output.data.rename(str(adj_var) , "a Function from dolfin-adjoint")
            else:
                output.data.name = str(adj_var)

        storage = libadjoint.MemoryStorage(output)
        storage.set_overwrite(True)
        adjglobals.adjointer.record_variable(adj_var, storage)

        # forget is None: forget *nothing*.
        # forget is True: forget everything we can, forward and adjoint
        # forget is False: forget only unnecessary adjoint values
        if forget is None:
            pass
        elif forget:
            adjglobals.adjointer.forget_adjoint_equation(i)
        else:
            adjglobals.adjointer.forget_adjoint_values(i)

        yield (output.data, adj_var)

def compute_tlm(parameter, forget=False):

    if isinstance(parameter, (list, tuple)):
        parameter = ListControl(parameter)

    for i in range(adjglobals.adjointer.equation_count):
        (tlm_var, output) = adjglobals.adjointer.get_tlm_solution(i, parameter)
        if output.data:
            output.data.rename(str(tlm_var), "a Function from dolfin-adjoint")

        storage = libadjoint.MemoryStorage(output)
        storage.set_overwrite(True)
        adjglobals.adjointer.record_variable(tlm_var, storage)

        yield (output.data, tlm_var)

        # forget is None: forget *nothing*.
        # forget is True: forget everything we can, forward and adjoint
        # forget is False: forget only unnecessary tlm values
        if forget is None:
            pass
        elif forget:
            adjglobals.adjointer.forget_tlm_equation(i)
        else:
            adjglobals.adjointer.forget_tlm_values(i)


def compute_gradient(J, param, forget=True, ignore=[], callback=lambda var, output: None, project=False):
    flag = misc.pause_annotation()

    enlisted_controls = enlist(param)
    param = ListControl(enlisted_controls)
    dJdparam = enlisted_controls.__class__([None] * len(enlisted_controls))

    last_timestep = adjglobals.adjointer.timestep_count

    ignorelist = []
    for fn in ignore:
        if isinstance(fn, backend.Function):
            ignorelist.append(adjglobals.adj_variables[fn])
        elif isinstance(fn, str):
            ignorelist.append(libadjoint.Variable(fn, 0, 0))
        else:
            ignorelist.append(fn)

    for i in range(adjglobals.adjointer.timestep_count):
        adjglobals.adjointer.set_functional_dependencies(J, i)

    for i in range(adjglobals.adjointer.equation_count)[::-1]:
        fwd_var = adjglobals.adjointer.get_forward_variable(i)
        if fwd_var in ignorelist:
            info("Ignoring the adjoint equation for %s" % fwd_var)
            continue

        (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, J)

        callback(adj_var, output.data)

        storage = libadjoint.MemoryStorage(output)
        storage.set_overwrite(True)
        adjglobals.adjointer.record_variable(adj_var, storage)
        fwd_var = libadjoint.Variable(adj_var.name, adj_var.timestep, adj_var.iteration)

        out = param.equation_partial_derivative(adjglobals.adjointer, output.data, i, fwd_var)
        dJdparam = _add(dJdparam, out)

        if last_timestep > adj_var.timestep:
            # We have hit a new timestep, and need to compute this timesteps' \partial J/\partial m contribution
            out = param.functional_partial_derivative(adjglobals.adjointer, J, adj_var.timestep)
            dJdparam = _add(dJdparam, out)

        last_timestep = adj_var.timestep

        if forget is None:
            pass
        elif forget:
            adjglobals.adjointer.forget_adjoint_equation(i)
        else:
            adjglobals.adjointer.forget_adjoint_values(i)

    rename(J, dJdparam, param)

    misc.continue_annotation(flag)

    return postprocess(dJdparam, project, list_type=enlisted_controls)

def rename(J, dJdparam, param):
    if isinstance(dJdparam, list):
        [rename(J, dJdm, m) for (dJdm, m) in zip(dJdparam, param.controls)]
    elif isinstance(dJdparam, backend.Function):
        dJdparam.rename("d(%s)/d(%s)" % (str(J), str(param)), "a Function from dolfin-adjoint")

def project_test(func):
    if isinstance(func, backend.Function):
        V = func.function_space()
        u = backend.TrialFunction(V)
        v = backend.TestFunction(V)
        M = backend.assemble(backend.inner(u, v)*backend.dx)
        proj = backend.Function(V)
        backend.solve(M, proj.vector(), func.vector())
        return proj
    else:
        return func

def postprocess(dJdparam, project, list_type):
    if isinstance(dJdparam, list):
        return delist([postprocess(x, project, list_type) for x in dJdparam], list_type=list_type)
    else:
        if project:
            dJdparam = project_test(dJdparam)

        if isinstance(dJdparam, float):
            dJdparam = backend.Constant(dJdparam)

    return dJdparam

def hessian(J, m, warn=True):
    '''Choose which Hessian the user wants.'''
    with misc.annotations(False):
        H = BasicHessian(J, m, warn=warn)
    return H

class BasicHessian(libadjoint.Matrix):
    '''A basic implementation of the Hessian class that recomputes the tangent linear, adjoint and second-order adjoint
    equations on each action. Should be the slowest, but safest, with the lowest memory requirements.'''
    def __init__(self, J, m, warn=True):
        self.J = J

        self.enlisted_controls = enlist(m)
        self.m = ListControl(self.enlisted_controls)

        if warn:
            backend.info_red("Warning: Hessian computation is still experimental and is known to not work for some problems. Please Taylor test thoroughly.")

    def __call__(self, m_dot, project=False):
        flag = misc.pause_annotation()
        hess_action_timer = backend.Timer("Hessian action")

        m_p = self.m.set_perturbation(m_dot)
        last_timestep = adjglobals.adjointer.timestep_count

        m_dot = enlist(m_dot)
        Hm = []
        for m_dot_cmp in m_dot:
            if hasattr(m_dot_cmp, 'function_space'):
                Hm.append(backend.Function(m_dot_cmp.function_space()))
            elif isinstance(m_dot_cmp, float):
                Hm.append(0.0)
            else:
                raise NotImplementedError("Sorry, don't know how to handle this")

        tlm_timer = backend.Timer("Hessian action (TLM)")
        # run the tangent linear model
        for (tlm, tlm_var) in compute_tlm(m_p, forget=None):
            pass

        tlm_timer.stop()

        # run the adjoint and second-order adjoint equations.
        for i in range(adjglobals.adjointer.equation_count)[::-1]:
            adj_var = adjglobals.adjointer.get_forward_variable(i).to_adjoint(self.J)
            # Only recompute the adjoint variable if we do not have it yet
            try:
                adj = adjglobals.adjointer.get_variable_value(adj_var)
            except (libadjoint.exceptions.LibadjointErrorHashFailed, libadjoint.exceptions.LibadjointErrorNeedValue):
                adj_timer = backend.Timer("Hessian action (ADM)")
                adj = adjglobals.adjointer.get_adjoint_solution(i, self.J)[1]
                adj_timer.stop()

                storage = libadjoint.MemoryStorage(adj)
                adjglobals.adjointer.record_variable(adj_var, storage)

            adj = adj.data

            soa_timer = backend.Timer("Hessian action (SOA)")
            (soa_var, soa_vec) = adjglobals.adjointer.get_soa_solution(i, self.J, m_p)
            soa_timer.stop()
            soa = soa_vec.data

            def hess_inner(Hm, out):
                assert len(out) == len(Hm)
                for i in range(len(out)):
                    if out[i] is not None:
                        if isinstance(Hm[i], backend.Function):
                            Hm[i].vector().axpy(1.0, out[i].vector())
                        elif isinstance(Hm[i], float):
                            Hm[i] += out[i]
                        else:
                            raise ValueError, "Do not know what to do with this"
                return Hm

            func_timer = backend.Timer("Hessian action (derivative formula)")
            # now implement the Hessian action formula.
            out = self.m.equation_partial_derivative(adjglobals.adjointer, soa, i, soa_var.to_forward())
            Hm = hess_inner(Hm, out)

            out = self.m.equation_partial_second_derivative(adjglobals.adjointer, adj, i, soa_var.to_forward(), m_dot)
            Hm = hess_inner(Hm, out)

            if last_timestep > adj_var.timestep:
                # We have hit a new timestep, and need to compute this timesteps' \partial^2 J/\partial m^2 contribution
                last_timestep = adj_var.timestep
                out = self.m.functional_partial_second_derivative(adjglobals.adjointer, self.J, adj_var.timestep, m_dot)
                Hm = hess_inner(Hm, out)

            func_timer.stop()

            storage = libadjoint.MemoryStorage(soa_vec)
            storage.set_overwrite(True)
            adjglobals.adjointer.record_variable(soa_var, storage)

        for Hm_cmp in Hm:
            if isinstance(Hm_cmp, backend.Function):
                Hm_cmp.rename("d^2(%s)/d(%s)^2" % (str(self.J), str(self.m)), "a Function from dolfin-adjoint")

        misc.continue_annotation(flag)
        return postprocess(Hm, project, list_type=self.enlisted_controls)

    def action(self, x, y):
        assert isinstance(x.data, backend.Function)
        assert isinstance(y.data, backend.Function)

        Hm = self.__call__(x.data)
        y.data.assign(Hm)

    def eigendecomposition(self, **kwargs):
        '''Compute the eigendecomposition of the Hessian.'''

        params = {'solver': 'krylovschur',
                  'spectrum': 'largest magnitude',
                  'type': 'hermitian',
                  'monitor': True,
                  'n': 1}

        params.update(kwargs)

        # We take in the options in "DOLFIN" syntax (the same
        # as SLEPcEigenSolver. libadjoint uses a different
        # syntax for the same things. Here we translate.
        # Sorry for the confusion.

        pairs = {'method': 'solver',
                 'type': 'type',
                 'which': 'spectrum',
                 'monitor': 'monitor',
                 'neigenpairs': 'n'}

        options = {}
        for key in pairs:
            options[key] = params[pairs[key]]

        # OK! Now add the model input and output vectors.
        if len(self.m.controls) != 1:
            raise ValueError, "Sorry, only support single controls at the moment."
        data = adjlinalg.Vector(self.m.data()[0])
        options['input'] = data
        options['output'] = data

        eps = adjglobals.adjointer.compute_eps(self, options)

        retval = []
        for i in range(eps.ncv):
            (lamda, u) = eps.get_eps(i)
            u.data.rename("eigenvector %s of d^2(%s)/d(%s)^2" % (i, str(self.J), str(self.m)), "a Function from dolfin-adjoint")
            retval += [(lamda, u.data)]

        return retval

def _add(value, increment):
    # Add increment to value correctly taking into account None.

    if increment is None:
        return value
    elif value is None:
        return increment
    else:
        if isinstance(value, list) or isinstance(increment, list):
            assert isinstance(value, list) and isinstance(increment, list)
            return value.__class__([_add(val, inc) for (val, inc) in zip(value, increment)])

        else:
            return value+increment

class compute_gradient_tlm(object):
    '''Rather than compute the gradient of a functional all at once with
    the adjoint, compute its action with the tangent linear model. Useful
    for testing tangent linear models, and might be useful in future where
    you have many functionals and few parameters.'''
    def __init__(self, J, m, forget=True, callback=lambda var, output: None, project=False):
        self.J = J

        if isinstance(m, (list, tuple)):
            m = ListControl(m)

        self.m = m
        self.forget = forget
        self.cb = callback
        self.project = project

        self.inner_cache = {}

    def vector(self):
        return self

    def __float__(self):
        return self.inner(1.0)

    def __getitem__(self, i):
        assert isinstance(self.m, ListControl)
        return compute_gradient_tlm(self.J, self.m[i], self.forget, self.cb, self.project)

    def cached_inner(self, vec):
        # Check if we've seen a *multiple* of this vec before;
        # this is the typical case in Taylor testing
        for cachevec in self.inner_cache.keys():
            try:
                ratioarray = cachevec.array() / vec.array()
            except AttributeError:
                ratioarray = [cachevec / vec]

            bools = ratioarray == ratioarray[0]
            if not isinstance(bools, ndarray): bools = [bools] # agh, how ugly this is.

            ratio = ratioarray[0]
            if all(bools) and ratio != 0:
                #info_green("compute_gradient_tlm cache hit")
                return self.inner_cache[cachevec] / ratio

        #info_red("compute_gradient_tlm cache miss")
        self.inner_cache[vec] = self.real_inner(vec)
        return self.inner_cache[vec]

    def inner(self, vec):
        '''Compute the action of the gradient on the vector vec.'''
        def make_mdot(vec):
            if isinstance(self.m, FunctionControl):
                mdot = self.m.set_perturbation(backend.Function(self.m.data().function_space(), vec))
            elif isinstance(self.m, ConstantControl):
                mdot = self.m.set_perturbation(backend.Constant(vec))

            return mdot

        mdot = make_mdot(vec)

        grad = 0.0
        last_timestep = -1

        for (tlm, tlm_var) in compute_tlm(mdot, forget=self.forget):
            self.cb(tlm_var, tlm)
            fwd_var = tlm_var.to_forward()
            dJdu = adjglobals.adjointer.evaluate_functional_derivative(self.J, fwd_var)
            if dJdu is not None and tlm is not None:
                dJdu_vec = backend.assemble(dJdu.data)
                grad = _add(grad, dJdu_vec.inner(tlm.vector()))

            if last_timestep < tlm_var.timestep:
                out = self.m.functional_partial_derivative(adjglobals.adjointer, self.J, tlm_var.timestep)
                grad = _add(grad, out)

            last_timestep = tlm_var.timestep

        return grad
