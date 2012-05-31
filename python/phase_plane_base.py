from burst import LileyBase
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

model = LileyBase(params = params[sys.argv[1]])

run = model.run([0, 16.5])

without_transient = run.run([0, 50]).display(['phi_ei', 'phi_ee'])
without_transient.display(["h_i", "h_e"], fig = "3")
#show()
frozen = without_transient.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 30])

#without_transient.display(['i_ee', 'i_ei'], fig = "2")
#show()

#cont = frozen.searchForBifurcations('phi_ee', 'h_e', steps = 80000, maxStepSize = 1).display(fig = "2")
#cont.followHopf('H1').displayMinMax(fig = "2")


#without_transient.display(['h_e', 'h_i'], fig = "1")
#without_transient.display(['phi_ee', 'phi_ei'], fig = "2")
gc.collect()

show()
