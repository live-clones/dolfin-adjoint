# from os import path
# import subprocess
import pytest
import numpy as np
from dolfin_adjoint import *
from multimesh_poisson import *
# @pytest.mark.skip("Not supported by dolfin master yet")
# def test(request):
#     test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
#     test_dir = path.split(str(request.fspath))[0]
#     test_cmd = ["python", path.join(test_dir, test_file)]

#     handle = subprocess.Popen(test_cmd, cwd=test_dir)
#     assert handle.wait() == 0

def test_multimesh_poisson_cg_moola():
    moola = pytest.importorskip("moola", minversion="0.1")
    Ns = [33,65]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_moola(Ns[i],"CG", 1)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 2
    assert(rate>exp_rate-0.05)

def test_multimesh_poisson_dg_moola():
    moola = pytest.importorskip("moola", minversion="0.1")
    Ns = [33,65]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_moola(Ns[i],"DG", 0)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 1
    assert(rate>exp_rate-0.05)

@pytest.mark.skipif(True, 'Not working yet')
def test_multimesh_poisson_cg_scipy():
    moola = pytest.importorskip("scipy")
    Ns = [33,65]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_scipy(Ns[i],"CG", 1)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 2
    assert(rate>exp_rate-0.05)

def test_multimesh_poisson_dg_scipy():
    moola = pytest.importorskip("scipy")
    Ns = [33,65]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_scipy(Ns[i],"DG", 0)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 1
    assert(rate>exp_rate-0.05)
