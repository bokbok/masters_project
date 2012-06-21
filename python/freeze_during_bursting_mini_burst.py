from model import LileyWithBurst
from pylab import plot, show, figure

# USE With biphasic_burst parameter set!!

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper



path = os.path.dirname(os.path.abspath(__file__))
params = load(file(path + '/../parameterisations/parameterisations.yml', 'r'))

burst = LileyWithBurst(params = params[sys.argv[1]])
run = burst.run([0, 156.8])
run.displayPhasePlane2D('h_e', 'h_i', fig = "4")

frozen = run.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t', 'slow_e', 'slow_i'])
run2 = frozen.run([0, 50])
run2.display(['h_e', 'h_i', 'i_ee', 'i_ei'], fig = "2")
run2.displayPhasePlane2D('h_e', 'h_i', fig = "5")
show()
