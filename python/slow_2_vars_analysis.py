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


run = burst.run([0, 40])
frozen = run.freeze(['slow_e', 'slow_i', 'phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 10])#.display(['h_e', 'h_i'], fig = "2")
run.display(['h_e', 'h_i'])
run.display(['slow_e', 'slow_i'], fig = "9")

cont = frozen.searchForBifurcations('slow_i', 'h_e', steps=5000).display(fig = "7")
#cont.followHopf2('H1', 'slow_i', steps = 1000, maxStepSize = 1e-3).display(displayVar = 'slow_i', fig = "4")
#cont.followHopf2('H1', 'T_i', steps = 1000, maxStepSize = 1e-3).display(displayVar = 'T_i', fig = "5")
cont.followHopf2('H1', 'gamma_i', steps = 200, maxStepSize = 1e-3).display(displayVar = 'gamma_i', fig = "6")
cont.followHopf2('H1', 'tor_i', steps = 200, maxStepSize = 1e-3).display(displayVar = 'tor_i', fig = "11")


gc.collect()

show()
