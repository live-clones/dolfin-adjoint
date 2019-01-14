import dolfin
from . import solving

dolfin_genericmatrix_add = dolfin.cpp.la.GenericMatrix.__add__

def adjoint_genericmatrix_add(self, other):
    out = dolfin_genericmatrix_add(self, other)
    if hasattr(self, 'form') and hasattr(other, 'form'):
        out.form = self.form + other.form

    return out

dolfin.cpp.la.GenericMatrix.__add__ = adjoint_genericmatrix_add

dolfin_genericmatrix_mul = dolfin.cpp.la.GenericMatrix.__mul__

def adjoint_genericmatrix_mul(self, other):
    out = dolfin_genericmatrix_mul(self, other)
    if hasattr(self, 'form') and isinstance(other, dolfin.GenericVector):
        if hasattr(other, 'form'):
            out.form = dolfin.action(self.form, other.form)
        elif hasattr(other, 'function'):
            if hasattr(other, 'function_factor'):
                out.form = dolfin.action(other.function_factor*self.form, other.function)
            else:
                out.form = dolfin.action(self.form, other.function)

    return out

dolfin.cpp.la.GenericMatrix.__mul__ = adjoint_genericmatrix_mul

dolfin_genericmatrix_copy = dolfin.cpp.la.GenericMatrix.copy

def adjoint_genericmatrix_copy(self):
    out = dolfin_genericmatrix_copy(self)
    if hasattr(self, 'form'):
        out.form = self.form
    if hasattr(self, 'assemble_system'):
        out.assemble_system = self.assemble_system

    return out

dolfin.cpp.la.GenericMatrix.copy = adjoint_genericmatrix_copy
