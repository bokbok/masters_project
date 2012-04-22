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
#burst.displayNullclines('h_e', 'h_i')
#run = burst.run([0, 156.8])
#run = burst.run([0, 156.7925])
#run = burst.run([0, 156.7]).display(['h_e', 'h_i', 'i_ee', 'i_ei'], fig = "2")
run = burst.run([0, 50]).display(['h_e', 'h_i', 'i_ee', 'i_ei'], fig = "2")
#run.displayPhasePlane2D('h_e', 'h_i', fig = "3")
run.displayPhasePlane3D('slow_i', 'h_e', 'h_i', fig = "4")
run.displayPhasePlane3D('slow_i', 'slow_e', 'h_e', fig = "4")

frozen = run.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t', 'slow_e', 'slow_i'])
#run2 = frozen.run([0, 100])

#frozen = run.freeze(['slow_e', 'slow_i'])
#f = frozen.run([0, 100])
#f.display(['h_e', 'h_i', 'i_ee', 'i_ei'], fig = "2")

#cont = run2.searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 2000).display(fig = "5")

#cont2 = run2.searchForBifurcations('slow_i', 'i_ii', dir = '+', steps = 2000).display(fig = "6")

#cont = run2.searchForBifurcations('slow_e', 'h_i', dir = '+', steps = 2000).display(fig = "7")

#cont2 = run2.searchForBifurcations('slow_e', 'i_ii', dir = '+', steps = 2000).display(fig = "8")

#cont2.followSaddleNode('LP2').display(fig = '8')

#h1 = cont.follow('H1', 5000, dir = '+').display(fig = "1")
#pd1 = h1.follow('PD1', 1000, dir = '+').display(fig = "1")

#h1.showCycles(coords = ('h_e', 'h_i'), fig="3")
#cont.showCycles(coords = ('h_e', 'h_i'), point = "LP2", fig="3")
#cont.follow('LP1').display(fig = "1")
#run2.display(['slow_e', 'slow_i', 'h_e', 'h_i'], fig = "2")
#run.displayNullclines('h_e', 'h_i')
#run.display(['h_e', 'h_i', 'slow_e'], fig = "2")
show()
