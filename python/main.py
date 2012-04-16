from burst import LileyWithBurst
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
#freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t'])
run = burst.run([0, 160.7])
cont = run.freeze(['phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t', 'slow_e', 'slow_i']).searchForBifurcations('slow_i', 'h_i', dir = '+', steps = 500)
cont.display(fig = "1")
h1 = cont.follow('H1', 1000, dir = '+').display(fig = "1")

#h1.showCycles(coords = ('h_e', 'h_i'), point = "PD1", fig="3")
#run2.display(['slow_e', 'slow_i', 'h_e', 'h_i'], fig = "2")
run.display(['h_e', 'h_i'], fig = "2")
cont.showAll()



#burst.display()
