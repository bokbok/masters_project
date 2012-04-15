from PyDSTool import *
from pylab import plot, show


class LileyWithBurst:
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

    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0,
                'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0, 'slow_e' : 0, 'slow_i' : 0 }

    def __init__(self, params, ics = None, name="LileyBurstBase"):
        print "Here"
        self.params = params
        self.name = name
        self.equations = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t',
                           'phi_ei_t' : LileyWithBurst.phi_ei_tt, 'phi_ee_t' : LileyWithBurst.phi_ee_tt,
                           'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
                           'i_ee_t' : LileyWithBurst.i_ee_tt, 'i_ei_t' : LileyWithBurst.i_ei_tt, 'i_ie_t' : LileyWithBurst.i_ie_tt, 'i_ii_t' : LileyWithBurst.i_ii_tt,
                           'h_e' : LileyWithBurst.h_e_t, 'h_i' : LileyWithBurst.h_i_t, 'slow_e' : LileyWithBurst.slow_e_t, 'slow_i' : LileyWithBurst.slow_i_t }
        self.auxFunctions = { 'Y_e_h_e' : (['h_e'], LileyWithBurst.Y_e_h_e),
                              'Y_e_h_i' : (['h_i'], LileyWithBurst.Y_e_h_i),
                              'Y_i_h_e' : (['h_e'], LileyWithBurst.Y_i_h_e),
                              'Y_i_h_i' : (['h_i'], LileyWithBurst.Y_i_h_i),
                              's_e' : (['h'], LileyWithBurst.s_e),
                              's_i' : (['h'], LileyWithBurst.s_i)}

        if ics == None:
            self.ics = LileyWithBurst.zeroIcs
        else:
            self.ics = ics

    def run(self, timeRange):
        self.DSargs = args(varspecs = self.equations, fnspecs = self.auxFunctions, name=self.name)
        self.DSargs.tdomain = timeRange
        self.DSargs.algparams = {'init_step':0.00001, 'atol': 1e-12, 'rtol': 1e-13, 'max_pts' : 10000000}
        self.DSargs.checklevel = 2
        self.DSargs.tdata=timeRange
        self.DSargs.pars = self.params
        self.DSargs.ics = self.ics

        self.odeSystem = Radau_ODEsystem(self.DSargs)


        print "Running...."
        self.traj = self.odeSystem.compute('run')
        self.points = self.traj.sample()
        print "Done."

        #set up ICs based on end equilibrium of above run

        contIcs = {}

        for k, v in self.points.iteritems():
         contIcs[k] = v[len(v) - 1]


        print contIcs

        return LileyWithBurst(params = self.params, ics = contIcs, name = self.name)

    def freeze(self, vars):
        for var in vars:
            del self.equations[var]
            self.params[var] = self.ics[var]
            del self.ics[var]
        for var in vars:
            print var, self.params[var]
        self.name = self.name + "_freeze_" + str(hash("".join(vars)))
        return self


    def display(self):
        if self.points == None:
            raise Error("Not run")

        plot(self.points['t'], self.points['h_e'], label='h_e')
        plot(self.points['t'], self.points['h_i'], label='h_i')
        show()











