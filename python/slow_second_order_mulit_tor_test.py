from model import LileyWith2ndOrderSlow
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

start = 3
end = 10

for x in range(start, end):
    path = os.path.dirname(os.path.abspath(__file__))
    params = load(file(path + '/../parameterisations/parameterisations.yml', 'r'))

    param_set = params["alpha_exp_2"]

    param_set['tor_i'] = 10e-3 * x
    param_set['tor_e'] = 10e-3 * start
    param_set['weight_slow_i'] = param_set['weight_slow_i'] * 0.1
    param_set['weight_slow_e'] = 0.015

    model = LileyWith2ndOrderSlow(param_set)

    model.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t']).run([0, 30]).display(['h_i'], fig = str(x) + "0")
    model.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t']).run([0, 30]).display(['h_e'], fig = str(x) + "1")

show()