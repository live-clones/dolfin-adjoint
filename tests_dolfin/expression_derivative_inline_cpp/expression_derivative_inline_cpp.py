from dolfin import *
from dolfin_adjoint import *


# An expression that depends on a and b
#cpp_code = '''
#class MyCppExpression : public Expression
#{
#public:
#  MyCppExpression() : Expression(), a(0), b(0) { }
#
#  void eval(Array<double>& values, const Array<double>& x) const
#  {
#    const double dx = x[0] - a;
#    values[0] = dx*b;
#  }
#public:
#    double a, b;
#};'''


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    f = Expression("(a-x[0])*(a-x[0])*b*b", a=a, b=b)
    f.dependencies = [a, b]

    dfda = Expression("2*(a-x[0])*b", a=a, b=b)
    dfdb = Expression("2*b*(a-x[0])*(a-x[0])", a=a, b=b)

    f.user_defined_derivatives = {a: dfda, b: dfdb}

    taylor_test_expression(f, V)
