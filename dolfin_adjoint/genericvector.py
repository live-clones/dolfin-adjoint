import dolfin

dolfin_genericvector_neg = dolfin.GenericVector.__neg__

def adjoint_genericvector_neg(self):
    out = dolfin_genericvector_neg(self)
    if hasattr(self, 'form'):
        out.form = -self.form
    if hasattr(self, 'function'):
        out.function = self.function
        out.function_factor = -1.0

    return out

dolfin.GenericVector.__neg__ = adjoint_genericvector_neg
