from model import LileyBase
from pylab import plot, show, figure
import gc

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

run_params['phi_c'] = 0

model = LileyBase(params = run_params, timescale = "ms")
equib = model.run([0, 2000]).freeze(['phi_ei', 'phi_ei_t']).run([0, 2000])

cont = equib.searchForBifurcations('phi_ei', 'h_e', steps = int(sys.argv[3]), maxStepSize = 1).display(fig = "3")
show()
