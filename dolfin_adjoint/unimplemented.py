import dolfin

class NonlinearVariationalProblem(dolfin.NonlinearVariationalProblem):
  def __init__(self, *args, **kwargs):
    dolfin.info_red("Warning: NonlinearVariationalProblem is not currently annotated.")
    dolfin.NonlinearVariationalProblem.__init__(self, *args, **kwargs)

class LinearVariationalProblem(dolfin.LinearVariationalProblem):
  def __init__(self, *args, **kwargs):
    dolfin.info_red("Warning: LinearVariationalProblem is not currently annotated.")
    dolfin.LinearVariationalProblem.__init__(self, *args, **kwargs)

class PETScKrylovSolver(dolfin.PETScKrylovSolver):
  def solve(self, *args, **kwargs):
    dolfin.info_red("Warning: PETScKrylovSolver.solve is not currently annotated.")
    return dolfin.PETScKrylovSolver.solve(self, *args, **kwargs)
