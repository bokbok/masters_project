from model import SIRU3
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

bif_param = sys.argv[3]
steps = int(sys.argv[4])


ics = SIRU3.zeroIcs.copy()

ics['C_e'] = 1
ics['C_i'] = 1
ics['h_e'] = run_params['h_e_rest']
ics['h_i'] = run_params['h_i_rest']

run_params['g_e'] = 0
run_params['g_i'] = 0
run_params['fake_laplacian'] = 0

run_params['mus_i'] = 0
run_params['mus_e'] = 0


run_params['e_ie'] = 1
run_params['e_ii'] = 1


model = SIRU3(params = run_params, timescale = "ms", ics = ics)

state = model.run([0, 1])
state = state.freeze(['C_e', 'C_i']).run([0, 10000])
cont = state.searchForBifurcations(bif_param, 'h_e', steps = steps, maxStepSize = 1, bidirectional = False)

cont.display(fig = "4")
#cont.followLimitCycles('H1', steps = 3000).displayMinMax(fig = "4")

# cont.followHopf('H1', 'fake_laplacian', steps = 200, dir = '+').display(fig = "5", displayVar = 'fake_laplacian')
#cont.followHopf('H1', 'C_i', steps = 1000, dir = '+').display(fig = "6", displayVar = 'C_i')
#cont.followHopf('H1', 'C_e', steps = 1000, dir = '+').display(fig = "6", displayVar = 'C_e')

show()



