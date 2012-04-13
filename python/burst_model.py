from PyDSTool import *
from yaml import load, dump
from pylab import plot, show

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    print "No CLoader available"
    from yaml import Loader, Dumper

path = os.path.dirname(os.path.abspath(__file__))
params = load(file(path + '/../parameterisations/parameterisations.yml', 'r'))

modelParams = params[sys.argv[1]]

print "Using model params: " 
print modelParams

h_e_t='(1/tor_e) * (-(h_e - h_e_rest) + (Y_e_h_e(h_e) * i_ee) + (Y_i_h_e(h_e) * (i_ie)) + burst_e * slow_e)'
h_i_t='(1/tor_i) * (-(h_i - h_i_rest) + (Y_e_h_i(h_i) * i_ei) + (Y_i_h_i(h_i) * (i_ii)) + burst_i * slow_i)'

slow_e_t='(1/tor_slow) * (mu_slow_e * (h_e_rest - h_e) - nu_slow_e * slow_e)'
slow_i_t='(1/tor_slow) * (mu_slow_i * (h_i_rest - h_i) - nu_slow_i * slow_i)'

i_ee_tt='-2*gamma_ee * i_ee_t - (gamma_ee * gamma_ee) * i_ee + T_ee * gamma_ee * exp(1) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)' 
i_ei_tt='-2*gamma_ei * i_ei_t - (gamma_ei * gamma_ei) * i_ei + T_ei * gamma_ei * exp(1) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'
i_ie_tt='-2*gamma_ie * i_ie_t - (gamma_ie * gamma_ie) * i_ie + T_ie * gamma_ie * exp(1) * (N_beta_ie * s_i(h_i) + phi_ie + p_ie)'
i_ii_tt='-2*gamma_ii * i_ii_t - (gamma_ii * gamma_ii) * i_ii + T_ii * gamma_ii * exp(1) * (N_beta_ii * s_i(h_i) + phi_ii + p_ii)'

phi_ee_tt = '-2 * v * A_ee * phi_ee_t + (v * v) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee)'
phi_ei_tt = '-2 * v * A_ee * phi_ei_t + (v * v) * (A_ei * A_ee) * (N_alpha_ei * s_e(h_e) - phi_ei)'

s_e = 's_e_max / (1 + (1 - r_abs * s_e_max) * exp(-sqrt(2) * (h - mu_e) / sigma_e))'

s_i = 's_i_max / (1 + (1 - r_abs * s_i_max) * exp(-sqrt(2) * (h - mu_i) / sigma_i))'

Y_e_h_e = '(h_ee_eq - h_e) / abs(h_ee_eq - h_e_rest)'
Y_e_h_i = '(h_ei_eq - h_i) / abs(h_ei_eq - h_i_rest)'
Y_i_h_e = '(h_ie_eq - h_e) / abs(h_ie_eq - h_e_rest)'
Y_i_h_i = '(h_ii_eq - h_i) / abs(h_ii_eq - h_i_rest)'

auxFunctions = { 'Y_e_h_e' : (['h_e'], Y_e_h_e),
'Y_e_h_i' : (['h_i'], Y_e_h_i),
'Y_i_h_e' : (['h_e'], Y_i_h_e),
'Y_i_h_i' : (['h_i'], Y_i_h_i),
's_e' : (['h'], s_e),
's_i' : (['h'], s_i)
}


varspecs = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t', 
'phi_ei_t' : phi_ei_tt, 'phi_ee_t' : phi_ee_tt, 
'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
'i_ee_t' : i_ee_tt, 'i_ei_t' : i_ei_tt, 'i_ie_t' : i_ie_tt, 'i_ii_t' : i_ii_tt,
'h_e' : h_e_t, 'h_i' : h_i_t, 'slow_e' : slow_e_t, 'slow_i' : slow_i_t }

DSargs = args(varspecs = varspecs, fnspecs = auxFunctions, name="Burst_Model")
DSargs.tdomain = [0,300]
DSargs.algparams = {'init_step':0.00001, 'atol': 1e-12, 'rtol': 1e-13, 'max_pts' : 10000000}
DSargs.checklevel = 2
#DSargs.ics={'w':-1.0}
DSargs.tdata=[0, 300]
DSargs.pars = modelParams
DSargs.ics = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0, 'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0, 'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0, 'slow_e' : 0, 'slow_i' : 0 }

#odeSystem = Vode_ODEsystem(DSargs)
odeSystem = Radau_ODEsystem(DSargs)

print "Running...."
traj = odeSystem.compute('run')
points = traj.sample()
print "Done."

plot(points['t'], points['h_e'], label='h_e')
plot(points['t'], points['h_i'], label='h_i')
show()



