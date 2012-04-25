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
run = burst.run([0, 37])
#run = run.display(['h_e', 'h_i', 'slow_e','slow_i'], fig = "1")
run.displayPhasePlane2D('slow_i', 'h_i', fig = "3")
#run.displayPhasePlane2D('slow_e', 'h_e', fig = "2")

#run.displayPhasePlane3D('slow_i', 'h_e', 'h_i', fig = "4")
#run.displayPhasePlane3D('slow_i', 'i_ii', 'h_i', fig = "4")
#run.displayPhasePlane3D('slow_i', 'i_ie', 'h_i', fig = "5")

#run.displayPhasePlane3D('slow_e', 'i_ee', 'h_e', fig = "6")
#run.displayPhasePlane3D('slow_e', 'i_ei', 'h_e', fig = "7")

frozen = run.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t', 'slow_e', 'slow_i'])
#run2 = frozen.run([0, 10])

run = None

#frozen = run.freeze(['slow_e', 'slow_i'])
#f = frozen.run([0, 100])
#f.display(['h_e', 'h_i', 'i_ee', 'i_ei'], fig = "2")

print "Here"
cont1 = frozen.searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 2000).display(fig = "3")
#cont2 = frozen.searchForBifurcations('slow_e', 'h_e', dir = '+', steps = 2000).display(fig = "2")
#cont = frozen.searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 500).display(fig = "2")
print "There"


#print "Following..."
h1a = cont1.followHopf('H1', 6000, dir = '+').display(fig = '3')
h2a = cont1.followHopf('H2', 2000, dir = '+').display(fig = '3')
#h1b = cont2.followHopf('H1', 2000, dir = '+').display(fig = '2')
#h2b = cont2.followHopf('H2', 2000, dir = '+').display(fig = '2')

#print "Following PD"
#pd1 = h1.followHopf('PD1', 100, dir = '+')
#print "followed PD"
#h1.showCycles(coords = ('h_e', 'h_i'), fig="3")

#pd1.showCycles(coords = ('h_e', 'h_i'), fig="7")

#cont.showCycles(coords = ('h_e', 'h_i'), point = "LP2", fig="3")
#cont.follow('LP1').display(fig = "1")
#run2.display(['slow_e', 'slow_i', 'h_e', 'h_i'], fig = "2")
#run.displayNullclines('h_e', 'h_i')
#run.display(['h_e', 'h_i', 'slow_e'], fig = "2")
show()
