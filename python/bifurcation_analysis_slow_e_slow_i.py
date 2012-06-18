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
#run.displayPhasePlane2D('slow_i', 'h_e', fig = "2")

#run.run([0, 10]).displayPhasePlane2D('slow_i', 'slow_e', fig = "2")
print "Done"

gc.collect()
frozen = run.freeze(['slow_i', 'slow_e', 'phi_ei', 'phi_ei_t', 'phi_ee', 'phi_ee_t']).run([0, 10])
gc.collect()

print "Cont"
cont2f = frozen.searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 1500)
print "done - forward"
#cont2r = frozen.searchForBifurcations('slow_i', 'h_i', dir = '-', steps = 1000).display(displayVar='slow_e', fig = "1")
print "done - reverse"
gc.collect()


#h1r = cont2r.followHopf('H1', 10000, dir = '+').display(fig = '2')

h1b = cont2f.followHopfCD2('H1', 'slow_e', 20000, dir = '-')
gc.collect()
h2b = cont2f.followHopfCD2('H2', 'slow_e', 20000, dir = '-')

cont2f.display(displayVar='slow_e', fig = "1")
h1b.display(displayVar='slow_e', fig = '1')
h2b.display(displayVar='slow_e', fig = '1')

run.displayPhasePlane2D('slow_i', 'slow_e', fig = "1")
gc.collect()
run.displayPhasePlane2D('slow_i', 'slow_e', fig = "1")

show()