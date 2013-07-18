from model import SIRU2
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

ics = SIRU2.zeroIcs.copy()

ics['C_e'] = 1
ics['C_i'] = 1
ics['h_e'] = run_params['h_e_rest']
ics['h_i'] = run_params['h_i_rest']


if len(sys.argv) > 3:
    run_params['fake_laplacian'] = float(sys.argv[3])
if len(sys.argv) > 4:
    run_params['p_ee'] = float(sys.argv[4])

model = SIRU2(params = run_params, timescale = "ms", ics = ics)

cont = model.run([0, 1]).freeze(['C_e', 'C_i']).run([0, 10000]).searchForBifurcations('p_ee', 'h_e', steps = 1000, maxStepSize = 1, bidirectional = False).display(fig = "4")
#cont.followLimitCycles('H1', steps = 3000).displayMinMax(fig = "4")

#cont.followHopf('H1', 'fake_laplacian', steps = 200, dir = '-').display(fig = "5", displayVar = 'fake_laplacian')
#cont.followHopf('H1', 'C_i', steps = 200, dir = '+').display(fig = "6", displayVar = 'C_i')

show()



