from model import LileyWith2ndOrderSlow
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


burst = LileyWith2ndOrderSlow(params = params[sys.argv[1]])


run = burst.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t']).run([0, 7])
frozen = run.freeze(['slow_e', 'slow_i', 'slow_e_t', 'slow_i_t']).run([0, 10])

frozen.searchForBifurcations('tor_i', 'h_e', steps=10).display(fig = "7")

run.display(['h_e', 'h_i'])
run.display(['slow_e', 'slow_i'], fig = "9")

gc.collect()

show()
