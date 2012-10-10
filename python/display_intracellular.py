from model import LileySigmoidBurstPSPRes
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

model = LileySigmoidBurstPSPRes(params = params.values()[0], timescale = "ms")

linewidth = 1

if len(sys.argv) > 2 and sys.argv[2] == "save":
    linewidth = 0.3

equib = model.run([0, 10000]).run([0, 16000]).display(['h_e'], fig = "3", linewidth = linewidth, label= "mV").display(['T_ee_aug'], fig = "1", linewidth = linewidth).display(['T_ei_aug'], fig = "2", linewidth = linewidth)

if len(sys.argv) > 2 and sys.argv[2] == "save":
    figure("3")
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-he-intra.pdf", format = 'pdf')

    figure("2")
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-T_ee-intra.pdf", format = 'pdf')

    figure("1")
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-T_ei-intra.pdf", format = 'pdf')
else:
    show()