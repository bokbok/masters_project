from model import LileySigmoidBurstThalamic
from pylab import plot, show, figure
import numpy as np
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

run_params = params[params.keys()[0]]

run_params["gamma_ie"] *= np.float(sys.argv[2])
run_params["gamma_ii"] *= np.float(sys.argv[2])

run_params["T_ie"] *= np.float(sys.argv[3])
run_params["T_ii"] *= np.float(sys.argv[3])

run_params["tor_i"] *= np.float(sys.argv[4])

model = LileySigmoidBurstThalamic(params = run_params, timescale = "ms")

equib = model.run([0, 10000]).run([0, 40000]).display(['h_e'], fig = "3").display(['slow'], fig = "1").display(['phi_ee', 'phi_ei'], fig = "2")

#equib.displayPhasePlane3D('phi_ee', 'phi_ei', 'h_e')
if len(sys.argv) > 5:
    if sys.argv[5] == "save":
        print "Saving modified set"
        out = open(sys.argv[1] + "-thal-rev-mod-" + str(sys.argv[2]) + "-" + str(sys.argv[3]) + "-" + str(sys.argv[4]) + "-" + str(sys.argv[5]) + ".yml", 'w')

        dump({ sys.argv[2] : run_params }, out)
        out.close()
show()