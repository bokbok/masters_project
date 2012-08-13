from model import LileyWithSigmoidSingleSlow
from pylab import show
import gc

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper



path = os.path.dirname(os.path.abspath(__file__))
params = load(file(sys.argv[1], 'r'))

run_params = params[params.keys()[0]]

multiplier = 1
weight_multiplier = 1
g = run_params['g']
tor_multiplier = 1

if len(sys.argv) > 2:
    multiplier = float(sys.argv[2])

if len(sys.argv) > 3:
    weight_multiplier = float(sys.argv[3])

if len(sys.argv) > 4:
    g = float(sys.argv[4])

if len(sys.argv) > 5:
    tor_multiplier = float(sys.argv[5])

run_params['gamma_ie'] *= multiplier
run_params['gamma_ii'] *= multiplier

run_params['weight_slow_e'] *= weight_multiplier
run_params['weight_slow_i'] *= weight_multiplier

run_params['g'] = g
run_params['tor_slow'] *= tor_multiplier

burst = LileyWithSigmoidSingleSlow(params = run_params, timescale = "ms")


run = burst.run([0, 100000])

#base_run = base.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 100])

#base_run.display(['h_e', 'h_i'], fig = "4")
#run.run([0, 10]).performFFT('slow_i').display()
run.display(['h_e'])
run.display(['slow'], fig = "9")

show()
