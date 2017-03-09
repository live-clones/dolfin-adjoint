# from os import path
# import subprocess
import pytest
import numpy as np
from dolfin_adjoint import *
from multimesh_poisson import *

def test_multimesh_poisson_cg_moola():
    moola = pytest.importorskip("moola", minversion="0.1")
    Ns = [17,33]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_moola(Ns[i],"CG", 1)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 2
    assert(rate>exp_rate-0.05)

def test_multimesh_poisson_dg_moola():
    moola = pytest.importorskip("moola", minversion="0.1")
    Ns = [17,33]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_moola(Ns[i],"DG", 0)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 1
    assert(rate>exp_rate-0.05)

def test_multimesh_poisson_cg_scipy():
    scipy = pytest.importorskip("scipy", minversion="0.17")
    Ns = [17,33]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_scipy(Ns[i],"CG", 1)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 1.8
    assert(rate>exp_rate-0.05)

def test_multimesh_poisson_dg_scipy():
    scipy = pytest.importorskip("scipy", minversion="0.17")
    Ns = [17,33]
    h, control = np.zeros(2), np.zeros(2)
    for i in range(2):
        adj_reset()
        h[i],control[i] = solve_poisson_scipy(Ns[i],"DG", 0)
    rate = np.log(control[1]/control[0])/np.log(h[1]/h[0])
    exp_rate = 1
    assert(rate>exp_rate-0.05)
