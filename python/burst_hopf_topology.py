from model import LileyWithSigmoidSingleSlow
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
run_params['tor_slow'] = run_params['tor_e'] * 50
run_params['g'] = 0.7
run_params['weight_slow_e'] = 1
run_params['weight_slow_i'] = 1

model = LileyWithSigmoidSingleSlow(params = run_params, timescale = "ms")
equib = model.run([0, 2000]).freeze(['slow']).run([0, 5000])

cont = equib.searchForBifurcations('slow', 'h_e', steps = 1000, maxStepSize = 1).display(fig = "3")


hopf = cont.specialPointNames('H')

special = {}

if len(hopf) > 0:
    file = open("tmp/candidates-hopf.txt", "a")
    file.write(sys.argv[1] + "|||" + sys.argv[2] + "||hopf_count=" + str(len(hopf)) + "\n")
    for point in hopf:
        file.write(">> Hopf at slow=" + str(cont.specialPoints(point)['slow']) + "\n")

        hopfPoint = cont.followHopf(point, steps = 2000)
        specialPoints = hopfPoint.specialPointNames("LPC")
        specialPoints.extend(hopfPoint.specialPointNames("PD"))
        specialPoints.extend(hopfPoint.specialPointNames("NS"))

        for specialName in specialPoints:
            file.write(">> " + specialName + " at slow=" + str(hopfPoint.specialPoints(specialName)['slow']) + "\n")



#show()


