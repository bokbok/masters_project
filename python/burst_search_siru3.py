from model import SIRU3
from pylab import show

import os, sys, inspect
cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper


params = load(file(sys.argv[1], 'r'))

bif_var = sys.argv[2]

file = open("tmp/candidates-siru3-" + bif_var + ".txt", "w")


for param_set in params.keys():
    run_params = params[param_set]
    ics = SIRU3.zeroIcs.copy()

    ics['C_e'] = 1
    ics['C_i'] = 1
    ics['h_e'] = run_params['h_e_rest']
    ics['h_i'] = run_params['h_i_rest']

    run_params['g_e'] = 0
    run_params['g_i'] = 0
    run_params['fake_laplacian'] = 0

    run_params['mus_i'] = 0
    run_params['mus_e'] = 0


    run_params['e_ee'] = 0
    run_params['e_ei'] = 0
    run_params['e_ie'] = 1
    run_params['e_ii'] = 1

    try:
        model = SIRU3(params = run_params, timescale = "ms", ics = ics)

        state = model.run([0, 1])
        freeze = state.freeze(['C_e', 'C_i']).run([0, 10000])
        #freeze = state.freeze(['phi_ee', 'phi_ei']).run((0, 10000))
        print "Searching: " + param_set
        cont = freeze.searchForBifurcations(bif_var, 'h_e', steps = 1000, maxStepSize = 1, bidirectional = False)

        hopf = cont.specialPointNames('H')
        saddlenode = cont.specialPointNames('LP')
        print hopf
        print saddlenode
        if len(hopf) >= 1 and len(saddlenode) >= 1:
            print "Burst candidate!! "
            file.write(sys.argv[1] + "|||" + param_set + "||hopf_count=" + str(len(hopf)) + "||saddle_node_count=" + str(len(saddlenode)) + "\n")
            for point in hopf:
                print point
                file.write(">> Hopf at " + bif_var + "=" + str(cont.specialPoints(point)[bif_var]) + "\n")

            for point in saddlenode:
                file.write(">> saddlenode at " + bif_var + "=" + str(cont.specialPoints(point)[bif_var]) + "\n")

            file.flush()

    except Exception as e:
        print e
