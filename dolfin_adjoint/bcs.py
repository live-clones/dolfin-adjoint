import backend
from . import misc

if backend.__name__ == "firedrake":
    from backend import DirichletBC

    #just turn annotations off if g is computed with projection.project

    orig_function_arg = DirichletBC.function_arg
    def new_function_arg_setter(self, g):
        with misc.annotations(False):
            return orig_function_arg.fset(self, g)

    #monkey-patch

    DirichletBC.function_arg = property(fget=orig_function_arg.fget,
                                        fset=new_function_arg_setter,
                                        fdel=orig_function_arg.fdel)

