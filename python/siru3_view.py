from model import SIRU3
from pylab import show

import os, sys, inspect, errno, time
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

params = load(file(sys.argv[1], 'r'))
run_params = params[sys.argv[2]]

run_params['fake_laplacian'] = 0

i = 3
while len(sys.argv) > i:
    val = sys.argv[i].split('=')
    print val
    if len(val) == 2:
        run_params[val[0]] = float(val[1])
    i += 1

ics = SIRU3.zeroIcs

model = SIRU3(params = run_params, ics = ics, timescale = "ms")

run = model.run([0, 50000]).display(['C_e', 'C_i'], fig = "2").display(['h_e'], fig = "4")

run.displayPhasePlane3D('h_e', 'C_e', 'C_i')

if 'save' in sys.argv:
    save_params = { 'params' : run_params }
    dir = 'parameterisations/derived/' + sys.argv[1] + '/' + sys.argv[2]
    mkdir_p(dir)
    out = open(dir + '/' + str(time.time())+ '.yml', 'w')
    dump(save_params, out)

show()



