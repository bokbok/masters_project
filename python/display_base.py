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

model = LileyBase(params = run_params, timescale = "ms")

equib = model.freeze(['phi_ee', 'phi_ei']).run([0, 40000]).display(['h_e'], fig = "3")#.display(['phi_ee', 'phi_ei'], fig = "1")

#equib.displayPhasePlane3D('phi_ee', 'phi_ei', 'h_e')
show()