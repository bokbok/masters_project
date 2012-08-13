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

model = LileyBase(params = params[sys.argv[2]], timescale = "ms")
equib = model.run([0, 10000]).display(['h_e'])

show()