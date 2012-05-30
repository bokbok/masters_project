from burst import LileyWithBurst
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



path = os.path.dirname(os.path.abspath(__file__))
params = load(file(path + '/../parameterisations/parameterisations.yml', 'r'))

burst = LileyWithBurst(params = params[sys.argv[1]])

run = burst.run([0, 16.5])
without_transient = run.run([0, 100])
without_transient.display(['h_e', 'h_i'])
without_transient.displayPhasePlane3D('slow_e', 'h_e', 'h_i', fig = "2")
without_transient.displayPhasePlane3D('slow_i', 'h_e', 'h_i', fig = "3")
without_transient.displayPhasePlane3D('slow_i', 'slow_e', 'h_i', fig = "4")
without_transient.displayPhasePlane3D('slow_i', 'slow_e', 'h_e', fig = "5")
without_transient.displayPhasePlane2D('slow_i', 'slow_e', fig = "6")
gc.collect()

show()
