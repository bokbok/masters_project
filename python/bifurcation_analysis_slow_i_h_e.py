from model import LileyWithBurst
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

burst = LileyWithBurst(params = params['alpha'])

run = burst.run([0, 19.4])
gc.collect()
#show()
print "PP"
run.displayPhasePlane2D('slow_i', 'h_e', fig = "2")
print "Done"

gc.collect()
frozen = run.freeze(['slow_i', 'slow_e', 'phi_ei', 'phi_ei_t', 'phi_ee', 'phi_ee_t']).run([0, 10])
run = None
gc.collect()

print "Cont"
cont = frozen.searchForBifurcations('slow_i', 'h_e', dir = '+', steps = 2000).display(fig = "2")
print "done - forward"
gc.collect()


h1 = cont.followHopf('H1', 10000, dir = '+').displayMinMax(fig = '2')
h1.displayMinMax3D('slow_i', 'h_i', 'h_e')
print "slow_e=" + str(frozen.params['slow_e'])
print "slow_i=" + str(frozen.params['slow_i'])
show()
