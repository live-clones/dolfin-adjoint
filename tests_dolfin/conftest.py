import random
import dolfin
import dolfin_adjoint

default_params = dolfin.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    dolfin.parameters.update(default_params)

    # Reset adjoint state
    dolfin_adjoint.adj_reset()

    # Fix the seed to avoid random test failures due to slight tolerance variations
    random.seed(21)
