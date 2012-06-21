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

run = burst.run([0, 16.5])#.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t'])
run.displayPhasePlane2D('slow_e', 'h_e', fig = "2")
run.run([0, 10]).displayPhasePlane2D('slow_e', 'h_e', fig = "2")
gc.collect()
print "PP"
#run.displayPhasePlane2D('slow_e', 'h_e', fig = "2")
print "Done"

frozen = run.freeze(['slow_e', 'phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t']).run([0, 10])
run = None
gc.collect()
gc.collect()
gc.collect()

print "Cont"
cont2f = frozen.searchForBifurcations('slow_e', 'h_e', dir = '+', steps = 100).display(fig = "2")
print "done - forward"
gc.collect()
gc.collect()
gc.collect()
cont2r = frozen.searchForBifurcations('slow_e', 'h_e', dir = '-', steps = 150).display(fig = "2")
print "done - rev"


#h1r = cont2r.followHopf('H1', 2000, dir = '+').displayMinMax(fig = '2')

#h1b = cont2f.followHopf('H1', 50000, dir = '+').display(fig = '2')

show()
