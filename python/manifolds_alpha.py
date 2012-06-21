from model import LileyWithBurst
from display import Display
from pylab import plot, show, figure
from PyDSTool import restart
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

manifoldDisplay = Display('h_e', 'slow_i', 'slow_e', xrange=0.08, yrange=0.0015, zrange = 0.0015)
burst = LileyWithBurst(params = params['alpha'])
pp = burst.run([0, 100])

manifoldDisplay.add3DPhasePlane(pp)

for end in [20.5, 21.5]:
    for step in range(0, 6):

        run1 = burst.run([0, end + step * 0.1])
        gc.collect()

        gc.collect()
        frozen1 = run1.freeze(['slow_i', 'slow_e', 'phi_ei', 'phi_ei_t', 'phi_ee', 'phi_ee_t']).run([0, 10])
        run1 = None
        run2 = None
        gc.collect()


        print "Cont"
        print step
        cont_i1 = frozen1.searchForBifurcations('slow_i', 'h_e', dir = '+', steps = 5000)
        cont_e1 = frozen1.searchForBifurcations('slow_e', 'h_e', dir = '+', steps = 5000)
        print "done - forward"
        gc.collect()


        h1_i1 = cont_i1.followHopf('H1', 1000, dir = '+')
        h2_i1 = cont_i1.followHopf('H2', 1000, dir = '+')

        manifoldDisplay.addSliceZ(cont_i1, frozen1.params['slow_e'])
        if not h1_i1 == None:
            manifoldDisplay.addSliceMinMaxZ(h1_i1, frozen1.params['slow_e'])
        if not h2_i1 == None:
            manifoldDisplay.addSliceMinMaxZ(h2_i1, frozen1.params['slow_e'])

        h1_e1 = cont_e1.followHopf('H1', 1000, dir = '+')
        h2_e1 = cont_e1.followHopf('H2', 1000, dir = '+')

        manifoldDisplay.addSliceY(cont_e1, frozen1.params['slow_e'])
        if not h1_e1 == None:
            manifoldDisplay.addSliceMinMaxY(h1_e1, frozen1.params['slow_e'])
        if not h2_e1 == None:
            manifoldDisplay.addSliceMinMaxY(h2_e1, frozen1.params['slow_e'])

        print "1 slow_e=" + str(frozen1.params['slow_e'])
        print "1 slow_i=" + str(frozen1.params['slow_i'])
        restart()
        gc.collect()
show()
