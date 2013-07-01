from model import SIRU2
from pylab import show

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper


params = load(file(sys.argv[1], 'r'))
run_params = params[sys.argv[2]]

ics = SIRU2.zeroIcs.copy()

ics['C_e'] = 0
ics['C_i'] = 1
ics['h_e'] = run_params['h_e_rest']
ics['h_i'] = run_params['h_i_rest']

model = SIRU2(params = run_params, timescale = "ms", ics = ics)

cont = model.run([0, 1]).freeze(['C_e', 'C_i']).run([0, 50000]).display(['h_e']).searchForBifurcations('C_e', 'h_e', steps = 1000, maxStepSize = 1, bidirectional = False).display(fig = "4")
#cont.followHopf('H1', steps = 2000).displayMinMax(fig = '4')
cont.followHopfCD2('H1', 'C_i', steps = 500, dir = '-').display(fig = "3", displayVar = 'C_i')

show()



