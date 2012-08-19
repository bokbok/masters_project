from model import LileyWithDifferingV
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
run_params['v_ee'] = run_params['v'] #* float(sys.argv[3])
run_params['v_ei'] = run_params['v'] #* float(sys.argv[3])
run_params['A_ee'] *= float(sys.argv[4])
run_params['A_ei'] *= float(sys.argv[5])
#run_params['N_alpha_ei'] = run_params['N_alpha_ee']

model = LileyWithDifferingV(params = run_params, timescale = "ms")

equib = model.run([0, 60000]).display(['h_e'], fig = "3").display(['phi_ee', 'phi_ei'], fig = "1")

equib.displayPhasePlane3D('phi_ee', 'phi_ei', 'h_e')

if len(sys.argv) > 6:
    if sys.argv[6] == "save":
        out = open(sys.argv[1] + '-' + sys.argv[2] + "-phi-burst.yml", 'w')
        dump({ sys.argv[2] : run_params }, out)
        out.close()
else:
    show()