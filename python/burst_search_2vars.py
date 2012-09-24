from model import LileySigmoidBurst
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


params = load(file(sys.argv[1], 'r'))

run_params = params[sys.argv[2]]
run_params['tor_slow'] = run_params['tor_e'] * 50
run_params['g_e'] = 0.7
run_params['g_i'] = 0.7
run_params['weight_slow_e'] = float(sys.argv[3])
run_params['weight_slow_i'] = float(sys.argv[3])

model = LileySigmoidBurst(params = run_params, timescale = "ms")
equib = model.run([0, 40000]).display(['h_e'], fig = "3").display(['slow'], fig = "1")

if len(sys.argv) > 4:
    if sys.argv[4] == "save":
        out = open(sys.argv[1] + '-' + sys.argv[2] + "-burst.yml", 'w')
        dump({ sys.argv[2] : run_params }, out)
        out.close()
else:
    show()