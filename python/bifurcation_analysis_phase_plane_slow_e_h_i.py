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

burst = LileyWithBurst(params = params['alpha'])

run = burst.run([0, 19.363])
gc.collect()
#show()
print "PP"
run.displayPhasePlane2D('slow_e', 'h_i', fig = "2")

print "Done"

gc.collect()
frozen = run.freeze(['slow_e', 'slow_i', 'phi_ei', 'phi_ei_t', 'phi_ee', 'phi_ee_t']).run([0, 10])
run = None
gc.collect()

print "Cont"
cont = frozen.searchForBifurcations('slow_e', 'h_i', dir = '+', steps = 1000).display(fig = "2")
print "done - forward"

gc.collect()

h1r = cont.followHopf('H1', 3000, dir = '+').displayMinMax(fig = '2')
gc.collect()
#cont.followHopf2('H1', additionalFreeVar='slow_i', steps = 500, dir = '+').display(fig = '2')

show()
