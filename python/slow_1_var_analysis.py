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

run = burst.run([0, 46])
frozen = run.freeze(['slow'])
frozen.searchForBifurcations('slow', 'h_e', steps = 1000).display(fig = "7")

run.display(['h_e', 'h_i'])
run.display(['slow'], fig = "9")

gc.collect()

show()