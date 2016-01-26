from dolfin import *
from dolfin_adjoint import *


# An expression that depends on a and b
base_code = '''
class MyCppExpression : public Expression
{
public:
      std::shared_ptr<Constant> a;
      std::shared_ptr<Constant> b;
  MyCppExpression() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double a_ = (double) *a;
    double b_ = (double) *b;
    values[0] = EXPRESSION;
  }
};'''

cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_*a_")
da_cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_ - b_*b_*a_")
db_cpp_code = base_code.replace("EXPRESSION", "2*(x[0] - a_)*b_*a_")


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    f = Expression(cpp_code)
    f.a = a; f.b = b
    f.dependencies = [a, b]

    dfda = Expression(da_cpp_code)
    dfda.a = a; dfda.b = b

    dfdb = Expression(db_cpp_code)
    dfdb.a = a; dfdb.b = b

    f.user_defined_derivatives = {a: dfda, b: dfdb}

    taylor_test_expression(f, V)
