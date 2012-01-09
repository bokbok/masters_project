# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="matt"
__date__ ="$22/12/2011 9:12:43 PM$"


import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from numpy import sqrt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

class Solution:
    def __init__(self, h_e_rest, h_i_rest, gamma_e, gamma_i, 
                 N_beta_ee, N_beta_ei,
                 N_beta_ie, N_beta_ii,
                 N_alpha_ee, N_alpha_ei,
                 A_ee, A_ei, tor_e, tor_i, v, 
                 T_e, T_i, 
                 p_ee, p_ei, p_ie, p_ii, s_e_max, s_i_max, mu_e, mu_i, sigma_e, sigma_i, h_e_eq, h_i_eq):
        self.h_e_rest = h_e_rest
        self.h_i_rest = h_i_rest
        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.N_beta_ee = N_beta_ee
        self.N_beta_ei = N_beta_ei
        self.N_beta_ie = N_beta_ie
        self.N_beta_ii = N_beta_ii
        self.N_alpha_ee = N_alpha_ee
        self.N_alpha_ei = N_alpha_ei
        self.A_ee = A_ee
        self.A_ei = A_ei
        self.tor_e = tor_e
        self.tor_i = tor_i
        self.v = v
        self.T_e = T_e
        self.T_i = T_i
        self.p_ee = p_ee
        self.p_ei = p_ei
        self.p_ie = p_ie
        self.p_ii = p_ii
        self.s_e_max = s_e_max
        self.s_i_max = s_i_max
        self.mu_e = mu_e
        self.mu_i = mu_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.h_e_eq = h_e_eq
        self.h_i_eq = h_i_eq
        
        self.gamma_e_2 = gamma_e * gamma_e
        self.gamma_i_2 = gamma_i * gamma_i
        self.v_2 = v * v
        self.A_ee_2 = A_ee * A_ee
        self.A_ei_2 = A_ei * A_ei

        self.T = 0.05
        self.MAXSTEPS = 5000
        self.X_max = 1
        self.X_steps = 50
        
        self.delta_x = float(self.X_max) / float(self.X_steps)
        self.delta_x_2 = self.delta_x * self.delta_x
        
    def run_ode(self):
        self.t = np.linspace(0, self.T, 1000)
        self.x = np.linspace(0, self.X_max, self.X_steps)
        initial_conditions = np.zeros(14 * self.X_steps)
        for step in range(0, self.X_steps):
            initial_conditions[8 + 14 * step] = self.h_e_rest
            initial_conditions[9 + 14 * step] = self.h_i_rest
            
        step_size = float(self.T)/float(self.MAXSTEPS)
        self._sol = odeint(lambda vals, t: self.system(vals, t), initial_conditions, self.t, mxstep=500, atol = 1e-2, rtol =1e-2)

    def show_results(self):
        h_e = []
        for item in self._sol:
           h_e.append( (item[8], item[8 + (self.X_steps / 2) * 14]) )
           #h_e.append( item[0] )
        
        plt.plot(self.t, h_e)
        plt.xlabel('t')
        plt.ylabel('h_e')
        plt.axis('normal')
        plt.title('Blah')
        plt.show()

    def system(self, vals, t):
        state = []

        gamma_e = self.gamma_e
        gamma_i = self.gamma_i
        T_e = self.T_e
        T_i = self.T_i
        p_ee = self.p_ee
        p_ei = self.p_ei
        p_ie = self.p_ie
        p_ii = self.p_ii
        tor_e = self.tor_e
        tor_i = self.tor_i
        h_e_rest = self.h_e_rest
        h_i_rest = self.h_i_rest
        v = self.v
        A_ee = self.A_ee
        A_ei = self.A_ei
        N_beta_ee = self.N_beta_ee
        N_beta_ei = self.N_beta_ei
        N_beta_ie = self.N_beta_ie
        N_beta_ii = self.N_beta_ii
        N_alpha_ee = self.N_alpha_ee
        N_alpha_ei = self.N_alpha_ei
        v_2 = self.v_2
        gamma_e_2 = self.gamma_e_2
        gamma_i_2 = self.gamma_i_2
        A_ee_2 = self.A_ee_2
        A_ei_2 = self.A_ei_2
        print t
        # 14
        for x_step in range(0, self.X_steps):
            p, q, i_ee, i_ei, r, s, i_ie, i_ii, h_e, h_i, z_e, z_i, phi_e, phi_i = self.extract_state_for(x_step, vals)

            phi_e_xx = self.calc_phi_e_xx(vals, x_step)
            phi_i_xx = self.calc_phi_i_xx(vals, x_step)
            #print phi_e_xx, phi_i_xx

            p_t = (-2 * gamma_e * p) - (gamma_e_2 * i_ee) + (T_e * gamma_e * exp(1) * ((N_beta_ee * self.s_e(h_e)) + phi_e + p_ee))
            q_t = (-2 * gamma_e * q) - (gamma_e_2 * i_ei) + (T_e * gamma_e * exp(1) * ((N_beta_ei * self.s_e(h_e)) + phi_i + p_ei))

            r_t = (-2 * gamma_i * r) - (gamma_i_2 * i_ie) + (T_i * gamma_i * exp(1) * ((N_beta_ie * self.s_i(h_i)) + p_ie))
            s_t = (-2 * gamma_i * s) - (gamma_i_2 * i_ii) + (T_i * gamma_i * exp(1) * ((N_beta_ii * self.s_i(h_i)) + p_ii))


            h_e_t = (1/tor_e) * ((h_e_rest - h_e) + (self.Y_e(h_e) * i_ee) + (self.Y_i(h_e) * i_ie))
            h_i_t = (1/tor_i) * ((h_i_rest - h_i) + (self.Y_e(h_i) * i_ei) + (self.Y_i(h_i) * i_ii))

            #print h_e_t, i_ee, ((h_e_rest - h_e) + (self.Y_e(h_e) * i_ee) + (self.Y_i(h_e) * i_ie))

            z_e_t = (-2 * v * A_ee * z_e) - (v_2 * A_ee_2 * phi_e) + (v_2 * phi_e_xx) + (v_2 * A_ee_2 * N_alpha_ee * self.s_e(h_e))
            z_i_t = (-2 * v * A_ei * z_i) - (v_2 * A_ei_2 * phi_i) + (v_2 * phi_i_xx) + (v_2 * A_ei_2 * N_alpha_ei * self.s_e(h_e))

            state.append(p_t)
            state.append(q_t)
            state.append(p)
            state.append(q)
            state.append(r_t)
            state.append(s_t)
            state.append(r)
            state.append(s)
            state.append(h_e_t)
            state.append(h_i_t)
            state.append(z_e_t)
            state.append(z_i_t)
            state.append(z_e)
            state.append(z_i)
        return state
    def extract_state_for(self, x_step, vals):
        if x_step >= 0 and x_step < self.X_steps:
            return vals[(x_step * 14):((x_step + 1)*14)]
        else:
            return np.zeros(14)
    def s_e(self, h_e):
        return self.s(self.s_e_max, h_e, self.mu_e, self.sigma_e)
    def s_i(self, h_i):
        return self.s(self.s_i_max, h_i, self.mu_i, self.sigma_i)
    def s(self, max, h, mu, sigma):
        return max / (1 + exp(- sqrt(2) * (h - mu) / sigma))
    def Y_e(self, h_e):
        return self.Y(h_e, self.h_e_eq, self.h_i_rest)
    def Y_i(self, h_i):
        return self.Y(h_i, self.h_i_eq, self.h_e_rest)
    def Y(self, h, h_eq, h_rest):
        return (h_eq - h) / abs(h_eq - h_rest)
    def calc_phi_e_xx(self, vals, x_step):
        return self.calc_phi_xx(vals, x_step, 12)
    def calc_phi_i_xx(self, vals, x_step):
        return self.calc_phi_xx(vals, x_step, 13)
    
    def calc_phi_xx(self, vals, x_step, phi_index):
        x_minus = self.extract_x_val(x_step - 1, vals, phi_index)
        x_plus = self.extract_x_val(x_step + 1, vals, phi_index)
        x = self.extract_x_val(x_step, vals, phi_index)
        return (x_minus - 2 * x + x_plus) / self.delta_x_2        
        
    def extract_x_val(self, x_step, vals, index):
        x = self.extract_state_for(x_step, vals)[index]
        if math.isnan(x):
            x = 0
        return x
        
    
if __name__ == "__main__":
    print "Running...."
    
# h_e_rest, h_i_rest, gamma_e, gamma_i, 
# N_beta_ee, N_beta_ei,
# N_beta_ie, N_beta_ii,
# N_alpha_ee, N_alpha_ei,
# A_ee, A_ei, tor_e, tor_i, v, 
# T_e, T_i, 
# p_ee, p_ei, p_ie, p_ii, 
# s_e_max, s_i_max, mu_e, mu_i, 
# sigma_e, sigma_i, h_e_eq, h_i_eq
    
    s = Solution(-70, -70, 300, 65, 
                 3034, 3034, 
                 536, 536, 
                 4000, 2000, 
                 0.4, 0.4, 0.01, 0.01, 700, 
                 0.4, 0.8, 
                 0, 0, 0, 0, 
                 500, 500, -50, -50, 
                 5, 5, 45, -90)
    s.run_ode()
    s.show_results()
    
