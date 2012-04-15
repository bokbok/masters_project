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
run2 = burst.run([0, 17]).freeze(['slow_e', 'slow_i', 'phi_ee', 'phi_ei', 'phi_ee_t', 'phi_ei_t'])
run2.run([0, 10])
run2.display()

#burst.display()
