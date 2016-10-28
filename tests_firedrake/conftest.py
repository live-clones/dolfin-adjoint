import random
import firedrake
import firedrake_adjoint

default_params = firedrake.parameters.copy()


def pytest_runtest_setup(item):
    firedrake.parameters.update(default_params)

    firedrake_adjoint.adj_reset()

    random.seed(21)
