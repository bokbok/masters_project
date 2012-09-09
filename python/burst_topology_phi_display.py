from model import LileyConstPhi
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
run_params['phi_r'] = 1
model = LileyConstPhi(params = run_params, timescale = "ms")
equib = model.run([0, 2000])

cont = equib.searchForBifurcations('phi_c', 'h_e', steps = int(sys.argv[3]), maxStepSize = 1).display(fig = "1")

hopf = cont.specialPointNames('H')
saddlenode = cont.specialPointNames('LP')

for bif in saddlenode:
    if cont.specialPoints(bif)['phi_c'] >= 0:
        add = cont.followHopfCD2(bif, 'phi_r', steps = 5000).display(fig = "1").display("phi_r", fig = "3")


for bif in hopf:
    if cont.specialPoints(bif)['phi_c'] >= 0:
        #cont.followHopf(bif, steps= 20000).displayMinMax(fig = "1")
        cont.followHopf2(bif, 'phi_r', steps = 5000).display("phi_r", fig = "3")


show()
