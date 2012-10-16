from model import LileySigmoidBurstThalamic
from pylab import plot, show, figure, savefig, ylim
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

run_params = params.values()[0]
model = LileySigmoidBurstThalamic(params = run_params, timescale = "ms")

linewidth = 1

if len(sys.argv) > 2 and sys.argv[2] == "save":
    linewidth = 0.3

equib = model.run([0, 30000]).run([0, 16000]).display(['h_e'], fig = "3", label= "mV", linewidth = linewidth, yrange=[-75, -45]).display(['slow'], fig = "1", linewidth = linewidth, multiplier = run_params['thal_e']).display(['slow'], fig = "2", linewidth = linewidth, multiplier = run_params['thal_i'])
equib.performFFT('h_e').display(fig = "11")

if len(sys.argv) > 2 and sys.argv[2] == "save":
    figure("3")
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-he-thal.pdf", format = 'pdf')
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-he-thal.png", format = 'png')
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-he-thal.eps", format = 'eps')

    figure("1")
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-slow-thal.pdf", format = 'pdf')
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-slow-thal.png", format = 'png')
    savefig("papers/frontiers-2012-images-revised/" + os.path.basename(sys.argv[1]).replace('.', '_') + "-slow-thal.eps", format = 'eps')
else:
    show()