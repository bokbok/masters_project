from model import SIRU2
from pylab import show

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper


params = load(file(sys.argv[1], 'r'))
run_params = params[sys.argv[2]]

run_params['fake_laplacian'] = float(sys.argv[3])

model = SIRU2(params = run_params, timescale = "ms")

cont = model.run([0, 50000]).display(['h_e'], fig = "4").display(['C_e', 'C_i'], fig = "5")

show()



