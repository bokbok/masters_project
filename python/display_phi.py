from model import LileyBase
from pylab import plot, show, figure, savefig
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

model = LileyBase(params = params.values()[0], timescale = "ms")

linewidth = 1

if len(sys.argv) > 2 and sys.argv[2] == "save":
    linewidth = 0.1

equib = model.run([0, 10000]).run([0, 16000]).display(['h_e'], fig = "3", label= "mV", linewidth = linewidth).display(['phi_ee'], fig = "1", linewidth = linewidth).display(['phi_ei'], fig = "2", linewidth = linewidth)

if len(sys.argv) > 2 and sys.argv[2] == "save":
    figure("3")
    savefig("papers/frontiers-2012-images/" + os.path.basename(sys.argv[1]) + "-he-phi.eps", format = 'eps')

    figure("1")
    savefig("papers/frontiers-2012-images/" + os.path.basename(sys.argv[1]) + "-phi_ee-phi.eps", format = 'eps')
    figure("2")
    savefig("papers/frontiers-2012-images/" + os.path.basename(sys.argv[1]) + "-phi_ei-phi.eps", format = 'eps')
else:
    show()