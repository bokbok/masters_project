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

run = burst.run([0, 16.5])
gc.collect()
#show()
print "PP"
run.displayPhasePlane2D('slow_e', 'h_i', fig = "2")

print "Done"

gc.collect()
frozen = run.freeze(['slow_e', 'phi_ei', 'phi_ei_t', 'phi_ee', 'phi_ee_t']).run([0, 10])
run = None
gc.collect()

print "Cont"
cont2f = frozen.searchForBifurcations('slow_e', 'h_i', dir = '+', steps = 3000).display(fig = "2")
print "done - forward"
gc.collect()
cont2r = frozen.searchForBifurcations('slow_e', 'h_i', dir = '-', steps = 50).display(fig = "2")
print "done - rev"


#h1r = cont2r.followHopf('H1', 10000, dir = '+').display(fig = '2')

#h1b = cont2f.followHopf('H1', 50000, dir = '+').display(fig = '2')

show()
