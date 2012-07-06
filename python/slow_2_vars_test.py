from model import LileyWithBurstSimplifiedParamSpace, LileyBase
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


burst = LileyWithBurstSimplifiedParamSpace(params = params[sys.argv[1]])
#base = LileyBase(params = params[sys.argv[1]])


run = burst.run([0, 300])

#base_run = base.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 100])

#base_run.display(['h_e', 'h_i'], fig = "4")
#run.run([0, 10]).performFFT('slow_i').display()
run.display(['h_e', 'h_i'])
run.display(['slow_e', 'slow_i'], fig = "9")

run.displayPhasePlane3D('h_e', 'slow_e', 'slow_i', fig = "7")

gc.collect()

show()
