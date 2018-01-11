import pytest
import importlib
import random
import dolfin
import dolfin_adjoint

@pytest.fixture(autouse=True)
def skip_by_missing_module(request):
    if request.node.get_marker('skipif_module_is_missing'):
        to_import = request.node.get_marker('skipif_module_is_missing').args[0]
    try:
        importlib.import_module(to_import)
    except ImportError:
            pytest.skip('skipped because module {} is missing'.format(to_import))   

default_params = dolfin.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    dolfin.parameters.update(default_params)

    # Reset adjoint state
    dolfin_adjoint.adj_reset()

    # Fix the seed to avoid random test failures due to slight tolerance variations
    random.seed(21)
