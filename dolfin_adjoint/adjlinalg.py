import libadjoint
import backend
import ufl
import adjglobals
import os
import os.path
import misc
import caching
import compatibility
import utils

class Vector(libadjoint.Vector):
    '''This class implements the libadjoint.Vector abstract base class for dolfin-adjoint.
    In particular, it must implement the data callbacks for tasks such as adding two vectors
    together, duplicating vectors, taking norms, etc., that occur in the process of constructing
    the adjoint equations.'''

    def __init__(self, data, zero=False, fn_space=None):

        self.data = data
        if not (self.data is None or isinstance(self.data, backend.Function) or
                isinstance(self.data, ufl.Form) or isinstance(self.data,
                    backend.MultiMeshFunction)):
            backend.error("Got " + str(self.data.__class__) + " as input to the Vector() class. Don't know how to handle that.")

        # self.zero is true if we can prove that the vector is zero.
        if data is None:
            self.zero = True
        else:
            self.zero = zero

        if fn_space is not None:
            self.fn_space = fn_space

    def duplicate(self):

        if isinstance(self.data, ufl.form.Form):
            # The data type will be determined by the first addto.
            data = None
        elif isinstance(self.data, backend.Function):
            try:
                fn_space = self.data.function_space().collapse()
            except:
                fn_space = self.data.function_space()
            data = backend.Function(fn_space)

        elif isinstance(self.data, backend.MultiMeshFunction):
            try:
                fn_space = self.data.function_space().collapse()
            except:
                fn_space = self.data.function_space()
            data = backend.MultiMeshFunction(fn_space)

        else:
            data = None

        fn_space = None
        if hasattr(self, "fn_space"):
            fn_space = self.fn_space

        return Vector(data, zero=True, fn_space=fn_space)

    def axpy(self, alpha, x):

        if hasattr(x, 'nonlinear_form'):
            self.nonlinear_form = x.nonlinear_form
            self.nonlinear_u = x.nonlinear_u
            self.nonlinear_bcs = x.nonlinear_bcs
            self.nonlinear_J = x.nonlinear_J

        if x.zero:
            return

        if (self.data is None):
            # self is an empty form.
            if isinstance(x.data, backend.Function):
                self.data = backend.Function(x.data)
                self.data.vector()._scale(alpha)
            if isinstance(x.data, backend.MultiMeshFunction):
                self.data = backend.MultiMeshFunction(x.data.function_space(),
                        x.data.vector())
                self.data.vector()._scale(alpha)
            else:
                self.data=alpha*x.data

        elif x.data is None:
            pass
        elif isinstance(self.data, backend.Coefficient):
            if isinstance(x.data, backend.Coefficient):
                self.data.vector().axpy(alpha, x.data.vector())
            else:
                # This occurs when adding a RHS derivative to an adjoint equation
                # corresponding to the initial conditions.
                self.data.vector().axpy(alpha, backend.assemble(x.data))
                self.data.form = alpha * x.data
        elif isinstance(x.data, ufl.form.Form) and isinstance(self.data, ufl.form.Form):

            # Let's do a bit of argument shuffling, shall we?
            xargs = ufl.algorithms.extract_arguments(x.data)
            sargs = ufl.algorithms.extract_arguments(self.data)

            if xargs != sargs:
                # OK, let's check that all of the function spaces are happy and so on.
                for i in range(len(xargs)):
                    assert xargs[i].element() == sargs[i].element()
                    assert xargs[i].function_space() == sargs[i].function_space()

                # Now that we are happy, let's replace the xargs with the sargs ones.
                x_form = backend.replace(x.data, dict(zip(xargs, sargs)))
            else:
                x_form = x.data

            self.data+=alpha*x_form
        elif isinstance(self.data, ufl.form.Form) and isinstance(x.data, backend.Function):
            #print "axpy assembling FormFunc. self.data is a %s; x.data is a %s" % (self.data.__class__, x.data.__class__)
            x_vec = x.data.vector().copy()
            self_vec = backend.assemble(self.data)
            self_vec.axpy(alpha, x_vec)
            new_fn = backend.Function(x.data.function_space())
            new_fn.vector()[:] = self_vec
            self.data = new_fn
            self.fn_space = self.data.function_space()
        elif isinstance(self.data, backend.MultiMeshFunction):
            raise NotImplementedError

        else:
            print "self.data.__class__: ", self.data.__class__
            print "x.data.__class__: ", x.data.__class__
            assert False

        self.zero = False

    def norm(self):

        if isinstance(self.data, backend.Function):
            return (abs(backend.assemble(backend.inner(self.data, self.data)*backend.dx)))**0.5
        elif isinstance(self.data, ufl.form.Form):
            return backend.assemble(self.data).norm("l2")
        elif isinstance(self.data, backend.MultiMeshFunction):
            raise NotImplementedError

    def dot_product(self,y):

        if isinstance(self.data, ufl.form.Form):
            return backend.assemble(backend.inner(self.data, y.data)*backend.dx)
        elif isinstance(self.data, backend.Function):
            if isinstance(y.data, ufl.form.Form):
                other = backend.assemble(y.data)
            else:
                other = y.data.vector()
            return self.data.vector().inner(other)
        else:
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to dot anything else.")

    def set_random(self):
        if isinstance(self.data, backend.MultiMeshFunction):
            raise NotImplementedError

        assert isinstance(self.data, backend.Function) or hasattr(self, "fn_space")
        if self.data is None:
            self.data = backend.Function(self.fn_space)

        vec = self.data.vector()
        for i in range(len(vec)):
            vec[i] = random.random()

        self.zero = False

    def size(self):
        if hasattr(self, "fn_space") and self.data is None:
            self.data = backend.Function(self.fn_space)

        if isinstance(self.data, backend.Function):
            return self.data.vector().local_size()

        if isinstance(self.data, ufl.form.Form):
            return backend.assemble(self.data).local_size()

        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to get the size.")

    def set_values(self, array):
        if isinstance(self.data, backend.MultiMeshFunction):
            raise NotImplementedError
        if isinstance(self.data, backend.Function):
            vec = self.data.vector()
            vec.set_local(array)
            vec.apply("insert")

            self.zero = False
        elif self.data is None and hasattr(self, 'fn_space'):
            self.data = backend.Function(self.fn_space)
            vec = self.data.vector()
            vec.set_local(array)
            self.zero = False
        else:
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to set values.")

    def get_values(self, array):
        if isinstance(self.data, backend.Function):
            vec = self.data.vector()
            try:
                vec.get_local(array)
            except NotImplementedError:
                array[:] = vec.get_local()

        else:
            raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to get values.")

    def write(self, var):
        filename = str(var)
        suffix = "h5"

        # Change naming scheme for compatibility with HDF5File routines
        filename = filename.replace(":","-")

        #if not os.path.isfile(filename+".%s" % suffix):
        #  backend.info_red("Warning: Overwriting checkpoint file "+filename+"."+suffix)
        file = backend.HDF5File(backend.mpi_comm_world(), filename+".%s" % suffix, "w")
        file.write(self.data, filename)
        file.close()

        # Save the function space into adjglobals.checkpoint_fs. It will be needed when reading the variable back in.
        adjglobals.checkpoint_fs[filename] = self.data.function_space()

    @staticmethod
    def read(var):
        filename = str(var)
        filename = filename.replace(":","-")
        suffix = "h5"

        V = adjglobals.checkpoint_fs[filename]
        v = backend.Function(V)
        file = backend.HDF5File(backend.mpi_comm_world(), filename+".%s" % suffix, "r")
        file.read(v, filename)
        file.close()

        return Vector(v)

    @staticmethod
    def delete(var):
        try:
            filename = str(var)
            filename = filename.replace(":","-")
            suffix = "h5"

            import os
            os.remove(filename+".%s" % suffix)
        except OSError:
            pass

class Matrix(libadjoint.Matrix):
    '''This class implements the libadjoint.Matrix abstract base class for the Dolfin adjoint.
    In particular, it must implement the data callbacks for tasks such as adding two matrices
    together, duplicating matrices, etc., that occur in the process of constructing
    the adjoint equations.'''

    def __init__(self, data, bcs=None, solver_parameters=None, adjoint=None, cache=False):

        if bcs is None:
            self.bcs = []
        else:
            self.bcs=bcs

        self.data = data

        if solver_parameters is not None:
            self.solver_parameters = solver_parameters
        else:
            self.solver_parameters = {}

        self.cache = cache

    def assemble_data(self):
        assert not isinstance(self.data, IdentityMatrix)
        if not self.cache:
            return backend.assemble(self.data)
        else:
            if self.data in caching.assembled_adj_forms:
                if backend.parameters["adjoint"]["debug_cache"]:
                    backend.info_green("Got an assembly cache hit")
                return caching.assembled_adj_forms[self.data]
            else:
                if backend.parameters["adjoint"]["debug_cache"]:
                    backend.info_red("Got an assembly cache miss")
                if hasattr(self.data.arguments()[0], '_V_multi'):
                    M = backend.assemble_multimesh(self.data)
                else:
                    M = backend.assemble(self.data)
                caching.assembled_adj_forms[self.data] = M
                return M

    def basic_solve(self, var, b):
        if isinstance(self.data, IdentityMatrix):
            x=b.duplicate()
            x.axpy(1.0, b)
            if isinstance(x.data, ufl.Form):
                x = Vector(backend.Function(x.fn_space, backend.assemble(x.data)))
        else:
            if var.type in ['ADJ_TLM', 'ADJ_ADJOINT', 'ADJ_SOA']:
                dirichlet_bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, backend.DirichletBC)]
                other_bcs  = [bc for bc in self.bcs if not isinstance(bc, backend.DirichletBC)]
                bcs = dirichlet_bcs + other_bcs
            else:
                bcs = self.bcs

            test = self.test_function()

            #FIXME: This is a hack, the test and trial function should carry
            # the MultiMeshFunctionSpace as an attribute
            if hasattr(test, '_V_multi'):
                x = Vector(backend.MultiMeshFunction(test._V_multi))
            else:
                x = Vector(backend.Function(test.function_space()))

            #print "b.data is a %s in the solution of %s" % (b.data.__class__, var)
            if b.data is None and not hasattr(b, 'nonlinear_form'):
                # This means we didn't get any contribution on the RHS of the adjoint system. This could be that the
                # simulation ran further ahead than when the functional was evaluated, or it could be that the
                # functional is set up incorrectly.
                backend.warning("Warning: got zero RHS for the solve associated with variable %s" % var)
            elif isinstance(b.data, backend.Function):

                assembled_lhs = self.assemble_data()
                [bc.apply(assembled_lhs) for bc in bcs]
                assembled_rhs = compatibility.assembled_rhs(b)
                [bc.apply(assembled_rhs) for bc in bcs]

                wrap_solve(assembled_lhs, x.data, assembled_rhs, self.solver_parameters)
            else:
                if hasattr(b, 'nonlinear_form'): # was a nonlinear solve
                    x = compatibility.assign_function_to_vector(x, b.nonlinear_u, function_space = test.function_space())
                    F = backend.replace(b.nonlinear_form, {b.nonlinear_u: x.data})
                    J = backend.replace(b.nonlinear_J, {b.nonlinear_u: x.data})
                    try:
                        compatibility.solve(F == 0, x.data, b.nonlinear_bcs, J=J, solver_parameters=self.solver_parameters)
                    except RuntimeError as rte:
                        x.data.vector()[:] = float("nan")

                else:
                    assembled_lhs = self.assemble_data()
                    [bc.apply(assembled_lhs) for bc in bcs]
                    assembled_rhs = wrap_assemble(b.data, test)
                    [bc.apply(assembled_rhs) for bc in bcs]

                    wrap_solve(assembled_lhs, x.data, assembled_rhs, self.solver_parameters)

        return x

    def caching_solve(self, var, b):
        if isinstance(self.data, IdentityMatrix):
            output = b.duplicate()
            output.axpy(1.0, b)
            if isinstance(output.data, ufl.Form):
                output = Vector(backend.Function(output.fn_space, backend.assemble(output.data)))
        elif b.data is None:
            backend.warning("Warning: got zero RHS for the solve associated with variable %s" % var)
            output = Vector(backend.Function(self.test_function().function_space()))
        else:
            dirichlet_bcs = [utils.homogenize(bc) for bc in self.bcs if isinstance(bc, backend.DirichletBC)]
            other_bcs  = [bc for bc in self.bcs if not isinstance(bc, backend.DirichletBC)]
            bcs = dirichlet_bcs + other_bcs

            output = Vector(backend.Function(self.test_function().function_space()))
            #print "b.data is a %s in the solution of %s" % (b.data.__class__, var)
            if backend.parameters["adjoint"]["symmetric_bcs"] and backend.__version__ > '1.2.0':
                assembler = backend.SystemAssembler(self.data, b.data, bcs)
                assembled_rhs = backend.Vector()
                assembler.assemble(assembled_rhs)
            elif isinstance(b.data, ufl.Form):
                assembled_rhs = wrap_assemble(b.data, self.test_function())
            else:
                if backend.__name__ == "dolfin":
                    assembled_rhs = b.data.vector()
                else:
                    assembled_rhs = b.data
            [bc.apply(assembled_rhs) for bc in bcs]

            if not var in caching.lu_solvers:
                if backend.parameters["adjoint"]["debug_cache"]:
                    backend.info_red("Got a cache miss for %s" % var)

                if backend.parameters["adjoint"]["symmetric_bcs"] and backend.__version__ > '1.2.0':
                    assembled_lhs = backend.Matrix()
                    assembler.assemble(assembled_lhs)
                else:
                    assembled_lhs = self.assemble_data()
                    [bc.apply(assembled_lhs) for bc in bcs]

                caching.lu_solvers[var] = compatibility.LUSolver(assembled_lhs, "mumps")
                caching.lu_solvers[var].parameters["reuse_factorization"] = True
            else:
                if backend.parameters["adjoint"]["debug_cache"]:
                    backend.info_green("Got a cache hit for %s" % var)

            caching.lu_solvers[var].solve(output.data.vector(), assembled_rhs)

        return output

    def solve(self, var, b):
        if backend.parameters["adjoint"]["cache_factorizations"] and var.type != "ADJ_FORWARD":
            x = self.caching_solve(var, b)
        else:
            x = self.basic_solve(var, b)

        return x

    def action(self, x, y):
        assert isinstance(x.data, backend.Function)
        assert isinstance(y.data, backend.Function)

        action_form = backend.action(self.data, x.data)
        action_vec  = backend.assemble(action_form)
        y.data.vector()[:] = action_vec

    def axpy(self, alpha, x):
        assert isinstance(x.data, ufl.Form)
        assert isinstance(self.data, ufl.Form)

        # Let's do a bit of argument shuffling, shall we?
        xargs = ufl.algorithms.extract_arguments(x.data)
        sargs = ufl.algorithms.extract_arguments(self.data)

        if xargs != sargs:
            # OK, let's check that all of the function spaces are happy and so on.
            for i in range(len(xargs)):
                assert xargs[i].element() == sargs[i].element()
                assert xargs[i].function_space() == sargs[i].function_space()

            # Now that we are happy, let's replace the xargs with the sargs ones.
            x_form = backend.replace(x.data, dict(zip(xargs, sargs)))
        else:
            x_form = x.data

        self.data+=alpha*x_form
        self.bcs += x.bcs # Err, I hope they are compatible ...
        self.bcs = misc.uniq(self.bcs)

    def test_function(self):
        '''test_function(self)

        Return the ufl.Argument corresponding to the trial space for the form'''

        return ufl.algorithms.extract_arguments(self.data)[-1]

class IdentityMatrix(object):
    '''Placeholder object for identity matrices'''
    pass

def wrap_solve(A, x, b, solver_parameters):
    '''Make my own solve, since solve(A, x, b) can't handle other solver_parameters
    like linear solver tolerances'''

    # Comment. Why does list_lu_solver_methods() not return, a, uhm, list?
    lu_solvers = ["lu", "mumps", "umfpack", "spooles", "superlu", "superlu_dist", "pastix", "petsc"]

    if backend.__name__ == "dolfin":
        # dolfin's API for expressing linear_solvers and preconditioners has changed in 1.4. Here I try
        # to support both.
        method = solver_parameters.get("linear_solver", "default")
        pc = solver_parameters.get("preconditioner", "default")

        if "nonlinear_solver" in solver_parameters or "newton_solver" in solver_parameters:
            nonlinear_solver = solver_parameters.get("nonlinear_solver", "newton")
            sub_options = nonlinear_solver + "_solver"

            if sub_options in solver_parameters:
                newton_options = solver_parameters[sub_options]

                method = newton_options.get("linear_solver", method)
                pc = newton_options.get("preconditioner", pc)

        # We get passed in a Function, turn it into a Vector
        x = x.vector()
        if method in lu_solvers or method == "default":
            if method == "lu": method = "default"
            solver = backend.LUSolver(method)

            if "lu_solver" in solver_parameters:
                solver.parameters.update(solver_parameters["lu_solver"])

            solver.solve(A, x, b)
            return
        else:
            solver = backend.KrylovSolver(method, pc)

            if "krylov_solver" in solver_parameters:
                solver.parameters.update(solver_parameters["krylov_solver"])

            solver.solve(A, x, b)
            return
    else:
        backend.solve(A, x, b, solver_parameters=solver_parameters)
        return

def wrap_assemble(form, test):
    '''If you do
       F = inner(grad(TrialFunction(V), grad(TestFunction(V))))
       a = lhs(F); L = rhs(F)
       solve(a == L, ...)

       it works, even though L is empty. But if you try to assemble(L) as we do here,
       you get a crash.

       This function wraps assemble to catch that crash and return an empty RHS instead.
    '''

    try:
        import pdb; pdb.set_trace()
        if hasattr(form.arguments()[0], '_V_multi'):
            b = backend.assemble_multimesh(form)
        else:
            b = backend.assemble(form)
    except RuntimeError:
        assert len(form.integrals()) == 0
        b = backend.Function(test.function_space()).vector()

    return b
