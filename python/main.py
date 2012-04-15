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
run2 = burst.run([0, 15]).freeze(['slow_e', 'slow_i', 'phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t'])
cont = run2.searchForBifurcations('slow_i', 'h_i', dir = '+')
cont.display()
cont.follow('H1', 10000, dir = '+').display()
cont.showAll()


#burst.display(['slow_e', 'slow_i', 'h_e', 'h_i'])

#burst.display()
