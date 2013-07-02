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

ics['C_e'] = 0.8
ics['C_i'] = 0.8
ics['h_e'] = run_params['h_e_rest']
ics['h_i'] = run_params['h_i_rest']


run_params['fake_laplacian'] = 0.01

model = SIRU2(params = run_params, timescale = "ms", ics = ics)

cont = model.run([0, 1]).freeze(['C_e', 'C_i']).run([0, 10000]).searchForBifurcations('fake_laplacian', 'h_e', steps = 5000, maxStepSize = 1, bidirectional = False).display(fig = "4")
#cont.followLimitCycle('H1', steps = 2000).displayMinMax(fig = '4')
#cont.followHopf('H1', 'C_i', steps = 10, dir = '+').display(fig = "3", displayVar = 'C_i')
#cont.followLP('LP1', 'C_i', steps = 50, dir = '-').display(fig = "3", displayVar = 'C_i')
#cont.followLP('LP1', 'C_i', steps = 50, dir = '+').display(fig = "3", displayVar = 'C_i')
#cont.followLP('LP1', 'fake_laplacian', steps = 50, dir = '+').display(fig = "5", displayVar = 'fake_laplacian')
#cont.followHopf('H1', 'fake_laplacian', steps = 50, dir = '+').display(fig = "5", displayVar = 'fake_laplacian')

show()



