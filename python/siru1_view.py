from model import SIRU1
from pylab import show

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper


params = load(file(sys.argv[1], 'r'))
run_params = params[sys.argv[2]]

ics = SIRU1.zeroIcs.copy()

ics['h_e'] = run_params['h_e_rest']
ics['h_i'] = run_params['h_i_rest']
ics['Ts_ee'] = run_params['T_ee']
ics['Ts_ei'] = run_params['T_ei']
ics['Ts_ie'] = run_params['T_ie']
ics['Ts_ii'] = run_params['T_ii']


run_params['fake_laplacian'] = float(sys.argv[3])

model = SIRU1(params = run_params, timescale = "ms", ics = ics)

cont = model.run([0, 50000]).display(['h_e'], fig = "4").display(['Ts_ee', 'Ts_ei', 'Ts_ie', 'Ts_ii'], fig = "5")

show()



