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

hopf_count = 1
saddle_node_count = 1

if len(sys.argv) > 2:
    hopf_count = int(sys.argv[3])

if len(sys.argv) > 3:
    saddle_node_count = int(sys.argv[4])

model = LileyBase(params = run_params, timescale = "ms")
equib = model.run([0, 2000]).run([0, 2000])

cont = equib.searchForBifurcations('phi_ei', 'h_e', steps = 2000, maxStepSize = 1).display(fig = "3")


hopf = cont.specialPointNames('H')
saddlenode = cont.specialPointNames('LP')
print hopf
print saddlenode
if len(hopf) >= hopf_count and len(saddlenode) >= saddle_node_count:
    print "Burst candidate!! "
    file = open("tmp/candidates-phi-ei.txt", "a")
    file.write(sys.argv[1] + "|||" + sys.argv[2] + "||hopf_count=" + str(len(hopf)) + "||saddle_node_count=" + str(len(saddlenode)) + "\n")
    for point in hopf:
        print point
        file.write(">> Hopf at phi_c=" + str(cont.specialPoints(point)['phi_ei']) + "\n")

    for point in saddlenode:
        file.write(">> saddlenode at phi_c=" + str(cont.specialPoints(point)['phi_ei']) + "\n")


#show()


