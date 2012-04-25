from burst import LileyWithBurst
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
run = burst.run([0, 37])
run.displayPhasePlane2D('slow_i', 'h_i', fig = "3")
#run.displayPhasePlane2D('slow_e', 'h_e', fig = "2")

frozen = run.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t', 'slow_e', 'slow_i'])

run = None

cont1 = frozen.searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 2000).display(fig = "3")
#cont2 = frozen.searchForBifurcations('slow_e', 'h_e', dir = '+', steps = 2000).display(fig = "2")


#print "Following..."
h1a = cont1.followHopf('H1', 12000, dir = '+').display(fig = '3')
h2a = cont1.followHopf('H2', 12000, dir = '+').display(fig = '3')
#h1b = cont2.followHopf('H1', 2000, dir = '+').display(fig = '2')
#h2b = cont2.followHopf('H2', 2000, dir = '+').display(fig = '2')

show()
