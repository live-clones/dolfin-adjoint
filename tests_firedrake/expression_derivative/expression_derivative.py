from firedrake import *
from firedrake_adjoint import *

class SourceExpression(Expression):
    def __init__(self, c, d, derivative=None, **kwargs):
        Expression.__init__(self)
        self.c = c
        self.d = d
        self.derivative = derivative

    def eval(self, value, x):

        if self.derivative is None:
            # Evaluate functional
            value[0] = self.c**2
            value[0] *= self.d

        elif self.derivative == self.c:
            # Evaluate derivative of functional wrt c
            value[0] = 2*self.c*self.d

        elif self.derivative == self.d:
            # Evaluate derivative of functional wrt d
            value[0] = self.c**2


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant(2)
    d = Constant(3)

    f = SourceExpression(c, d, degree=3)
    f.dependencies = c, d  # firedrake-adjoint needs to know on which
                           # coefficients this expression depends on

    # Provide the derivative coefficients
    f.user_defined_derivatives = {c: SourceExpression(c, d, derivative=c, degree=3),
                                  d: SourceExpression(c, d, derivative=d, degree=3)}
    taylor_test_expression(f, V)
