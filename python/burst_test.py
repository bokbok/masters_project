from burst import LileyWith2ndOrderSlow
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

run = burst.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 16.5])
without_transient = run.run([0, 10])
without_transient.display(['h_e', 'h_i'])
without_transient.display(['slow_i', 'slow_e'], fig = "9")

gc.collect()

show()
