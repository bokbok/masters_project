from model import LileyWithBurst
from pylab import plot, show, figure


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
run = burst.run([0, 100])
run.display(['h_e', 'h_i', 'slow_e', 'slow_i'], fig = "2")
run.displayPhasePlane3D('h_i', 'slow_e', 'slow_i', fig = "3")

show()

