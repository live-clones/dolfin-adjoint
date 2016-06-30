#!/usr/bin/env python

import os, os.path
import sys
import subprocess
import multiprocessing
import time
from optparse import OptionParser

test_cmds = {'tlm_simple': 'mpirun -n 2 python tlm_simple.py',
             'svd_simple': 'mpirun -n 2 python svd_simple.py',
             'gst_mass': 'mpirun -n 2 python gst_mass.py',
             'hessian_eps': 'python hessian_eps.py',
             'optimization_scipy': 'mpirun -n 2 python optimization_scipy.py',
             'optimization_checkpointing': 'python optimization_checkpointing.py',
             'optimization_optizelle': 'mpirun -n 2 python optimization_optizelle.py',
             'optimal_control_mms': 'mpirun -n 2 python optimal_control_mms.py',
             'preassembly_efficiency': 'mpirun -n 1 python preassembly_efficiency.py --ignore; mpirun -n 1 python preassembly_efficiency.py',
             'differentiability-dg-upwind': None,
             'differentiability-stokes': None,
             'checkpoint_online': None,
             'changing_vector': None,
             'matrix_free_burgers': None,
             'matrix_free_heat': None,
             'matrix_free_simple': None,
             'ode_tentusscher': None,
             'svd_burgers': None,
             'rush_larsen': None,
             'function_assigner': None,
             'mantle_convection': None,
             'nullspace': None}

parser = OptionParser()
parser.add_option("-n", type="int", dest="num_procs", default = 1, help = "To run on N cores, use -n N; to use all processors available, run test.py -n 0.")
parser.add_option("-t", type="string", dest="test_name", help = "To run one specific test, use -t TESTNAME. By default all test are run.")
parser.add_option("-s", dest="short_only", default = False, action="store_true", help = "To run the short tests only, use -s. By default all test are run.")
parser.add_option("--timings", dest="timings", default=False, action="store_true", help = "Print timings of tests.")
(options, args) = parser.parse_args(sys.argv)

if options.num_procs <= 0:
    options.num_procs = None

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
if options.test_name:
    if not options.test_name in subdirs:
        print "Specified test not found."
        sys.exit(1)
    else:
        subdirs = [options.test_name]

long_tests = ["viscoelasticity", "cahn_hilliard", "optimization_scipy", "svd_burgers_perturb", "supg", "mpec"] # special case the very long tests for speed
for test in long_tests:
    subdirs.remove(test)

# Keep path variables (for buildbot's sake for instance)
orig_pythonpath = os.getenv('PYTHONPATH', '')
pythonpath = os.pathsep.join([os.path.abspath(os.path.join(basedir, os.path.pardir)), orig_pythonpath])
os.putenv('PYTHONPATH', pythonpath)

timings = {}

def f(subdir):
    test_cmd = test_cmds.get(subdir, 'python %s.py' % subdir)
    if test_cmd is not None:

        print "--------------------------------------------------------"
        print "Running %s " % subdir
        print "--------------------------------------------------------"

        start_time = time.time()
        handle = subprocess.Popen(test_cmd, shell=True, cwd=os.path.join(basedir, subdir))
        exit = handle.wait()
        end_time   = time.time()
        timings[subdir] = end_time - start_time
        if exit != 0:
            print "subdir: ", subdir
            print "exit: ", exit
            return subdir
        else:
            return None

tests = sorted(subdirs)
if not options.short_only:
    tests = long_tests + tests

pool = multiprocessing.Pool(options.num_procs)

fails = pool.map(f, tests)
# Remove Nones
fails = [fail for fail in fails if fail is not None]

if options.timings:
    for subdir in sorted(timings, key=timings.get, reverse=True):
        print "%s : %s s" % (subdir, timings[subdir])

if len(fails) > 0:
    print "Failures: ", set(fails)
    sys.exit(1)
