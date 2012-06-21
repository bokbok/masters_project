from model import LileyWithSingle1stOrderSlow
from pylab import show
import gc

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper



path = os.path.dirname(os.path.abspath(__file__))
params = load(file(path + '/../parameterisations/parameterisations.yml', 'r'))


burst = LileyWithSingle1stOrderSlow(params = params[sys.argv[1]])
#base = LileyBase(params = params[sys.argv[1]])


run = burst.run([0, 100])

#base_run = base.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 100])

#base_run.display(['h_e', 'h_i'], fig = "4")

run.display(['h_e', 'h_i'])
run.display(['slow'], fig = "9")

gc.collect()

show()
