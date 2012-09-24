from model import LileySigmoidBurstPSP
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

model = LileySigmoidBurstPSP(params = params.values()[0], timescale = "ms")

linewidth = 1

if len(sys.argv) > 2 and sys.argv[2] == "save":
    linewidth = 0.1

equib = model.run([0, 10000]).run([0, 16000]).display(['h_e'], fig = "3", linewidth = linewidth).display(['slow'], fig = "1", linewidth = linewidth)

if len(sys.argv) > 2 and sys.argv[2] == "save":
    figure("3")
    savefig("papers/frontiers-2012-images/" + params.keys()[0] + "-he-intra.eps", format = 'eps')

    figure("1")
    savefig("papers/frontiers-2012-images/" + params.keys()[0] + "-slow-intra.eps", format = 'eps')
else:
    show()