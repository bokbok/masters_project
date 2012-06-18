from burst import LileyWithWeightedCoupling
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

model = LileyWithWeightedCoupling(params = params[sys.argv[1]])

run = model.freeze(['phi_ee', 'phi_ee_t', 'phi_ei', 'phi_ei_t']).run([0, 16.5])

without_transient = run.run([0, 30])

without_transient.display(['h_e', 'h_i'], fig = "1")
show()
#show()

cont_e = without_transient.searchForBifurcations('p_ii', 'h_e', steps = 1000000, maxStepSize = 1e-3).display(fig = "2")
h1 = cont_e.followHopf('H1', steps = 500)
if h1:
   h1.displayMinMax(fig = "2")

h2 = cont_e.followHopf('H2', steps = 500, maxStepSize = 1e-5)
if h2:
    h2.displayMinMax(fig = "2")
#
#cont_i = without_transient.searchForBifurcations('weight_i', 'h_e', steps = 20000).display(fig = "3")
#cont_i.followHopf('H1', steps = 1000, maxStepSize = 1).displayMinMax(fig = "3")

#without_transient.display(['h_e', 'h_i'], fig = "1")
#without_transient.display(['phi_ee', 'phi_ei'], fig = "2")

show()
