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

run_params['fake_laplacian'] = 0

i = 3
while len(sys.argv) > i:
    val = sys.argv[i].split('=')
    print val
    run_params[val[0]] = float(val[1])
    i += 1



model = SIRU2(params = run_params, timescale = "ms")

cont = model.run([0, 50000]).display(['h_e'], fig = "4").display(['C_e', 'C_i'], fig = "5").display(['phi_ee', 'phi_ei'], fig = "6")

show()



