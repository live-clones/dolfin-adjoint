from dolfin import *
from dolfin_adjoint import *

class SourceExpression(Expression):
    def __init__(self, c, d):
        self.c = c
        self.d = d

    def eval(self, value, x):
        value[0] = self.c**2
        value[0] *= self.d

    def deval(self, value, x, derivative_coeff):

        if self.c == derivative_coeff:
            value[0] = 2*self.c*self.d

        elif self.d == derivative_coeff:
            value[0] = self.c**2

    def dependencies(self):
        return [self.c, self.d]

    def copy(self):
        return SourceExpression(self.c, self.d)


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant(2)
    d = Constant(3)

    source = SourceExpression(c, d)
    taylor_test_expression(source, V)
