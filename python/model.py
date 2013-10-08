from PyDSTool import *
from pylab import plot, show, figure, draw, legend, xlabel, ylabel, ylim
from mpl_toolkits.mplot3d import Axes3D

from numpy import exp;

from analysis import FFT

class LileyBase(object):
    h_e_t='(1/tor_e) * (-(h_e - h_e_rest) + (Y_e_h_e(h_e) * i_ee) + (Y_i_h_e(h_e) * (i_ie)))'
    h_i_t='(1/tor_i) * (-(h_i - h_i_rest) + (Y_e_h_i(h_i) * i_ei) + (Y_i_h_i(h_i) * (i_ii)))'

    i_ee_tt='-2*gamma_ee * i_ee_t - (gamma_ee * gamma_ee) * i_ee + T_ee * gamma_ee * exp(1) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)'
    i_ei_tt='-2*gamma_ei * i_ei_t - (gamma_ei * gamma_ei) * i_ei + T_ei * gamma_ei * exp(1) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'
    i_ie_tt='-2*gamma_ie * i_ie_t - (gamma_ie * gamma_ie) * i_ie + T_ie * gamma_ie * exp(1) * (N_beta_ie * s_i(h_i) + phi_ie + p_ie)'
    i_ii_tt='-2*gamma_ii * i_ii_t - (gamma_ii * gamma_ii) * i_ii + T_ii * gamma_ii * exp(1) * (N_beta_ii * s_i(h_i) + phi_ii + p_ii)'

    phi_ee_tt = '-2 * v * A_ee * phi_ee_t + (v * v) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee)'
    phi_ei_tt = '-2 * v * A_ei * phi_ei_t + (v * v) * (A_ei * A_ei) * (N_alpha_ei * s_e(h_e) - phi_ei)'

    s_e = 's_e_max / (1 + (1 - r_abs * s_e_max) * exp(-sqrt(2) * (h - mu_e) / sigma_e))'

    s_i = 's_i_max / (1 + (1 - r_abs * s_i_max) * exp(-sqrt(2) * (h - mu_i) / sigma_i))'

    Y_e_h_e = '(h_ee_eq - h_e) / abs(h_ee_eq - h_e_rest)'
    Y_e_h_i = '(h_ei_eq - h_i) / abs(h_ei_eq - h_i_rest)'
    Y_i_h_e = '(h_ie_eq - h_e) / abs(h_ie_eq - h_e_rest)'
    Y_i_h_i = '(h_ii_eq - h_i) / abs(h_ii_eq - h_i_rest)'

    gamma = 'gamma_o * e_lk / (exp(e_lk) - 1)'
    gamma_tilde = '_gamma(gamma_o, e_lk) * exp(e_lk)'

    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0,
                'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0 }

    odeSystems = {}

    def __init__(self, params, ics = None, name="LileyBase", equations = None, points = None, odeSystem = None, auxFunctions = None, timescale = "s"):
        self.params = params
        self.name = name
        self.points = points
        self.timescale = self._convertTimescale(timescale)
        self.timescaleLabel = timescale
        print ics
        print self.name
        if equations == None:
            print "No eqns for " + name
            self.equations = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t',
                               'phi_ei_t' : LileyBase.phi_ei_tt, 'phi_ee_t' : LileyBase.phi_ee_tt,
                               'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
                               'i_ee_t' : LileyBase.i_ee_tt, 'i_ei_t' : LileyBase.i_ei_tt, 'i_ie_t' : LileyBase.i_ie_tt, 'i_ii_t' : LileyBase.i_ii_tt,
                               'h_e' : LileyBase.h_e_t, 'h_i' : LileyBase.h_i_t}
        else:
            self.equations = equations

        if auxFunctions == None:
            self.auxFunctions = { 'Y_e_h_e' : (['h_e'], LileyBase.Y_e_h_e),
                                  'Y_e_h_i' : (['h_i'], LileyBase.Y_e_h_i),
                                  'Y_i_h_e' : (['h_e'], LileyBase.Y_i_h_e),
                                  'Y_i_h_i' : (['h_i'], LileyBase.Y_i_h_i),
                                  's_e' : (['h'], LileyBase.s_e),
                                  's_i' : (['h'], LileyBase.s_i),
                                  '_gamma' : (['gamma_o', 'e_lk'], LileyBase.gamma),
                                  '_gamma_tilde' : (['gamma_o', 'e_lk'], LileyBase.gamma_tilde)}

        else:
            print auxFunctions
            self.auxFunctions = auxFunctions

        if ics == None:
            self.ics = LileyBase.zeroIcs
            self.ics['h_e'] = params['h_e_rest']
            self.ics['h_i'] = params['h_i_rest']
        else:
            self.ics = ics

        print self.ics
        if odeSystem == None:
            self.odes = self._odeSystem()
        else:
            self.odes = odeSystem

        print equations

    def performFFT(self, axis):
        deltaT = (self.points["t"][1] - self.points["t"][0]) / self.timescale

        fft = FFT(self.points[axis], deltaT, axis)
        fft.compute()

        return fft

    def run(self, timeRange):
        print "Running....", self.params
        self.odes.set(tdomain = timeRange)
        self.odes.set(tdata = timeRange)
        self.odes.set(ics = self.ics)
        self.odes.set(pars = self.params)
        traj = self.odes.compute('run')
        points = traj.sample()
        print "Done."

        #set up ICs based on end equilibrium of above run

        contIcs = {}

        for k, v in points.iteritems():
         contIcs[k] = v[len(v) - 1]

        return LileyBase(params = self.params, ics = contIcs, name = self.name, points = points, equations = self.equations, odeSystem = self.odes, auxFunctions = self.auxFunctions, timescale = self.timescaleLabel)

    def _odeSystem(self, timeRange = [0, 30]):
        if not LileyBase.odeSystems.has_key(self.name):
            DSargs = args(varspecs = self.equations, fnspecs = self.auxFunctions, name=self.name)
            DSargs.tdomain = [0, 3000]
            DSargs.algparams = {'init_step':2e-6,
                                'atol': 1e-12 / self.timescale,
                                'rtol': 1e-13 / self.timescale,
                                'max_pts' : 4000000}
            DSargs.checklevel = 2
            DSargs.tdata = timeRange
            DSargs.pars = self.params
            DSargs.ics = self.ics
            LileyBase.odeSystems[self.name] = Radau_ODEsystem(DSargs)

        return LileyBase.odeSystems[self.name]

    def freeze(self, vars):
        params = self.params.copy()
        ics = self.ics.copy()
        equations = self.equations.copy()

        for var in vars:
            del equations[var]
            params[var] = self.ics[var]
            del ics[var]
        name = self.name + "_freeze_" + str(abs(hash("-".join(vars))))
        return LileyBase(params = params, ics = ics, name = name, equations = equations, auxFunctions = self.auxFunctions)

    def searchForBifurcations(self, freeVar, displayVar, steps = 1000, dir = '+', maxStepSize = 1e-3, minStepSize = 1e-6, bidirectional = True):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        cont = ContClass(self.odes)

        contName = self.name + "_cont_" + freeVar
        PCargs = args(name=contName, type='EP-C')
        PCargs.freepars = [freeVar]
        PCargs.StepSize = 1e-6 / self.timescale
        PCargs.MaxNumPoints = steps
        PCargs.MaxStepSize = maxStepSize / self.timescale
        PCargs.MinStepSize = minStepSize / self.timescale
        PCargs.verbosity = 2
        PCargs.FuncTol = 1e-6 / self.timescale
        PCargs.VarTol = 1e-6 / self.timescale

        PCargs.LocBifPoints = 'all'

        # Declare a new curve based on the above criteria
        cont.newCurve(PCargs)

        # Do path following in the 'forward' direction. Max points is large enough
        # to ensure we go right around the ellipse (PyCont automatically stops when
        # we return to the initial point - unless MaxNumPoints is reached first.)
        if bidirectional:
            cont[contName].forward()
            cont[contName].backward()
        else:
            if dir == '+':
                cont[contName].forward()
            else:
                cont[contName].backward()


        sol = cont[contName].sol

        return Continuation(odeSystem = self.odes,
                            cont = cont,
                            sol = sol,
                            name = contName,
                            displayVar = displayVar,
                            freeVar = freeVar)

    def display(self, vars, fig = "1", label = "", linewidth = 1, yrange = None, multiplier = 1):
        if self.points == None:
            raise Error("Not run")

        for var in vars:
            figure(fig)
            if yrange:
                ylim(yrange)
            plot(self.points['t'] * self.timescale, self.points[var] * multiplier, label=var, linewidth = linewidth, color = 'black')
            xlabel(" t (s)")
            ylabel(label)
        return self

    def displayPhasePlane2D(self, x, y, fig = "1"):
        if self.points == None:
            raise Error("Not run")

        figure(fig)
        plot(self.points[x][0::10], self.points[y][0::10], label=x + " - " + y)
        return self

    def displayPhasePlane3D(self, x, y, z, fig = "5"):
        figr = figure(fig)
        axes = Axes3D(figr)
        axes.plot(self.points[x], self.points[y], self.points[z], label='3D')
        axes.legend()
        axes.set_xlabel(x)
        axes.set_ylabel(y)
        axes.set_zlabel(z)
        draw()
        return self

    def vals(self, axis):
        return self.points[axis]

    def _convertTimescale(self, timescale):
        if timescale == "ms":
            return 1e-3
        return 1


class LileyWithDifferingV(LileyBase):
    phi_ee_tt = '-2 * v_ee * A_ee * phi_ee_t + (v_ee * v_ee) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee)'
    phi_ei_tt = '-2 * v_ei * A_ei * phi_ei_t + (v_ei * v_ei) * (A_ei * A_ei) * (N_alpha_ei * s_e(h_e) - phi_ei)'


    def __init__(self, params, ics = None, name="LileyWithDifferingV", equations = None, points = None, odeSystem = None, timescale = "s"):
        if equations == None:
            print "No eqns for " + name
            equations = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t',
                               'phi_ei_t' : LileyWithDifferingV.phi_ei_tt, 'phi_ee_t' : LileyWithDifferingV.phi_ee_tt,
                               'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
                               'i_ee_t' : LileyBase.i_ee_tt, 'i_ei_t' : LileyBase.i_ei_tt, 'i_ie_t' : LileyBase.i_ie_tt, 'i_ii_t' : LileyBase.i_ii_tt,
                               'h_e' : LileyBase.h_e_t, 'h_i' : LileyBase.h_i_t}
        else:
            equations = equations

        if ics == None:
            ics = LileyBase.zeroIcs
            ics['h_e'] = params['h_e_rest']
            ics['h_i'] = params['h_i_rest']
            print "ICS.."
            print ics

        LileyBase.__init__(self, params = params, ics = ics, name = name, equations = equations, points = points, odeSystem = odeSystem, timescale = timescale)

class LileySigmoidBurstPSPRes(LileyBase):
    T_ee_aug_t='(T_ee - T_ee_aug - T_ee_aug * g_ee * s_e(h_e))/tor_slow_ee'
    T_ei_aug_t='(T_ei - T_ei_aug - T_ei_aug * g_ei * s_e(h_e))/tor_slow_ei'

    i_ee_tt='-2*gamma_ee * i_ee_t - (gamma_ee * gamma_ee) * i_ee + T_ee_aug * gamma_ee * exp(1) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)'
    i_ei_tt='-2*gamma_ei * i_ei_t - (gamma_ei * gamma_ei) * i_ei + T_ei_aug * gamma_ei * exp(1) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'


    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0,
                'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0 }

    def __init__(self, params, ics = None, name="LileySigmoidBurstPSPRes", equations = None, points = None, odeSystem = None, timescale = "s"):
        if equations == None:
            print "No eqns for " + name
            equations = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t',
                               'phi_ei_t' : LileyBase.phi_ei_tt, 'phi_ee_t' : LileyBase.phi_ee_tt,
                               'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
                               'i_ee_t' : LileySigmoidBurstPSPRes.i_ee_tt, 'i_ei_t' : LileySigmoidBurstPSPRes.i_ei_tt, 'i_ie_t' : LileyBase.i_ie_tt, 'i_ii_t' : LileyBase.i_ii_tt,
                               'h_e' : LileyBase.h_e_t, 'h_i' : LileyBase.h_i_t, 'T_ee_aug' : LileySigmoidBurstPSPRes.T_ee_aug_t, 'T_ei_aug' : LileySigmoidBurstPSPRes.T_ei_aug_t }
        else:
            equations = equations

        if ics == None:
            ics = LileySigmoidBurstPSPRes.zeroIcs
            ics['h_e'] = params['h_e_rest']
            ics['h_i'] = params['h_i_rest']
            ics['T_ee_aug'] = params['T_ee']
            ics['T_ei_aug'] = params['T_ei']
            print "ICS.."
            print ics

        LileyBase.__init__(self, params = params, ics = ics, name = name, equations = equations, points = points, odeSystem = odeSystem, timescale = timescale)


class SIRU1(LileyBase):
    Ts_ek_t = "mus_e * (theta_e - k_e * s_e(h_e))"
    Ts_ik_t = "mus_i * (theta_i - k_i * s_i(h_i))"

    i_ee_tt='-(_gamma_ee + gamma_ee_tilde) * i_ee_t - (_gamma_ee * gamma_ee_tilde) * i_ee + Ts_ee * gamma_ee_tilde * exp(_gamma_ee / gamma_ee) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)'
    i_ei_tt='-(_gamma_ei + gamma_ee_tilde) * i_ei_t - (_gamma_ei * gamma_ei_tilde) * i_ei + Ts_ei * gamma_ei_tilde * exp(_gamma_ei / gamma_ei) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'
    i_ie_tt='-(_gamma_ie + gamma_ie_tilde) * i_ie_t - (_gamma_ie * gamma_ie_tilde) * i_ie + Ts_ie * gamma_ie_tilde * exp(_gamma_ie / gamma_ie) * (N_beta_ie * s_i(h_i) + phi_ie + p_ie)'
    i_ii_tt='-(_gamma_ii + gamma_ii_tilde) * i_ii_t - (_gamma_ii * gamma_ii_tilde) * i_ii + Ts_ii * gamma_ii_tilde * exp(_gamma_ii / gamma_ii) * (N_beta_ii * s_i(h_i) + phi_ii + p_ii)'

    phi_ee_tt = '-2 * v * A_ee * phi_ee_t + (v * v) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee) + 1.5 * v * v * fake_laplacian'
    phi_ei_tt = '-2 * v * A_ei * phi_ei_t + (v * v) * (A_ei * A_ei) * (N_alpha_ei * s_e(h_e) - phi_ei) + 1.5 * v * v * fake_laplacian'

    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0,
                'phi_ei' : 0, 'phi_ei_t' : 0,
                'i_ee' : 0, 'i_ee_t' : 0, 'i_ei' : 0,
                'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0,
                'Ts_ee' : 0, 'Ts_ei' : 0, 'Ts_ie' : 0, 'Ts_ii' : 0}

    def __init__(self, params, ics = None, name="SIRU1", equations = None, points = None, odeSystem = None, timescale = "s"):
        if equations == None:
            print "No eqns for " + name
            equations = { 'phi_ee' : 'phi_ee_t',
                          'phi_ei' : 'phi_ei_t',
                          'phi_ei_t' : SIRU1.phi_ei_tt,
                          'phi_ee_t' : SIRU1.phi_ee_tt,
                          'i_ee' : 'i_ee_t',
                          'i_ei' : 'i_ei_t',
                          'i_ie' : 'i_ie_t',
                          'i_ii' : 'i_ii_t',
                          'i_ee_t' : SIRU1.i_ee_tt,
                          'i_ei_t' : SIRU1.i_ei_tt,
                          'i_ie_t' : SIRU1.i_ie_tt,
                          'i_ii_t' : SIRU1.i_ii_tt,
                          'h_e' : LileyBase.h_e_t,
                          'h_i' : LileyBase.h_i_t,
                          'Ts_ee' : SIRU1.Ts_ek_t,
                          'Ts_ei' : SIRU1.Ts_ek_t,
                          'Ts_ie' : SIRU1.Ts_ik_t,
                          'Ts_ii' : SIRU1.Ts_ik_t}
        else:
            equations = equations

        prepared = self.prepareParams(params)
        if ics == None:
            ics = SIRU1.zeroIcs
            ics['h_e'] = params['h_e_rest']
            ics['h_i'] = params['h_i_rest']
            ics['Ts_ee'] = params['T_ee']
            ics['Ts_ei'] = params['T_ei']
            ics['Ts_ie'] = params['T_ie']
            ics['Ts_ii'] = params['T_ii']
            print "ICS.."
            print ics

        LileyBase.__init__(self, params = prepared, ics = ics, name = name, equations = equations, points = points, odeSystem = odeSystem, timescale = timescale)

    def prepareParams(self, params):
        prepared = params.copy()

        if not 'fake_laplacian' in prepared:
            prepared['fake_laplacian'] = 0

        for dim in ['ee', 'ei', 'ie', 'ii']:
            param = 'gamma_' + dim
            paramMod = '_gamma_' + dim
            paramTilde = param + '_tilde'
            rate = 'e_' + dim

            orig = params[param]
            if params[rate] > 0:
                print rate + ": " + str(params[rate])
                prepared[paramMod] = (orig * params[rate]) / (exp(params[rate]) - 1)
                prepared[paramTilde] = prepared[paramMod] * exp(params[rate]);
            else:
                prepared[paramTilde] = orig
                prepared[paramMod] = orig

            print param + ": " + str(prepared[param])
            print paramMod + ": " + str(prepared[paramMod])
            print paramTilde + ": " + str(prepared[paramTilde])

        return prepared


class SIRU2(LileyBase):
    C_i_t = "mus_i * (1 / (1 + exp(k_i * (h_i - theta_i)))  - C_i)"
    C_e_t = "mus_e * (1 / (1 + exp(k_e * (h_e - theta_e)))  - C_e)"


    i_ee_tt='-(gamma_ee + gamma_ee) * i_ee_t - (gamma_ee * gamma_ee) * i_ee + (C_e * T_ee) * gamma_ee * exp(1) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)'
    i_ei_tt='-(gamma_ei + gamma_ei) * i_ei_t - (gamma_ei * gamma_ei) * i_ei + (C_e * T_ei) * gamma_ei * exp(1) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'
    i_ie_tt='-(_gamma(gamma_ie, e_ie) + _gamma_tilde(gamma_ie, e_ie)) * i_ie_t - (_gamma(gamma_ie, e_ie) * _gamma_tilde(gamma_ie, e_ie)) * i_ie + (C_i * T_ie) * _gamma_tilde(gamma_ie, e_ie) * exp(_gamma(gamma_ie, e_ie) / gamma_ie) * (N_beta_ie * s_i(h_i) + phi_ie + p_ie)'
    i_ii_tt='-(_gamma(gamma_ii, e_ii) + _gamma_tilde(gamma_ii, e_ii)) * i_ii_t - (_gamma(gamma_ii, e_ii) * _gamma_tilde(gamma_ii, e_ii)) * i_ii + (C_i * T_ii) * _gamma_tilde(gamma_ii, e_ii) * exp(_gamma(gamma_ii, e_ii) / gamma_ii) * (N_beta_ii * s_i(h_i) + phi_ii + p_ii)'

    phi_ee_tt = '-2 * v * A_ee * phi_ee_t + (v * v) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee) + 1.5 * v * v * fake_laplacian'
    phi_ei_tt = '-2 * v * A_ei * phi_ei_t + (v * v) * (A_ei * A_ei) * (N_alpha_ei * s_e(h_e) - phi_ei) + 1.5 * v * v * fake_laplacian'

    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0,
                'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0,
                'C_i' : 1, 'C_e' : 1}

    def __init__(self, params, ics = None, name="SIRU2", equations = None, points = None, odeSystem = None, timescale = "s"):
        if equations == None:
            print "No eqns for " + name
            equations = { 'phi_ee' : 'phi_ee_t',
                          'phi_ei' : 'phi_ei_t',
                          'phi_ei_t' : SIRU2.phi_ei_tt,
                          'phi_ee_t' : SIRU2.phi_ee_tt,
                          'i_ee' : 'i_ee_t',
                          'i_ei' : 'i_ei_t',
                          'i_ie' : 'i_ie_t',
                          'i_ii' : 'i_ii_t',
                          'i_ee_t' : SIRU2.i_ee_tt,
                          'i_ei_t' : SIRU2.i_ei_tt,
                          'i_ie_t' : SIRU2.i_ie_tt,
                          'i_ii_t' : SIRU2.i_ii_tt,
                          'h_e' : LileyBase.h_e_t,
                          'h_i' : LileyBase.h_i_t,
                          'C_e' : SIRU2.C_e_t,
                          'C_i' : SIRU2.C_i_t}
        else:
            equations = equations

        if ics == None:
            ics = SIRU2.zeroIcs
            ics['h_e'] = params['h_e_rest']
            ics['h_i'] = params['h_i_rest']
            print "ICS.."
            print ics

        LileyBase.__init__(self, params = params, ics = ics, name = name, equations = equations, points = points, odeSystem = odeSystem, timescale = timescale)


class SIRU3(LileyBase):
    C_e_t = 'mus_e * (1 - C_e * (1 + g_e * s_e(h_e)))'
    C_i_t = 'mus_i * (1 - C_i * (1 + g_i * s_i(h_i)))'

    i_ee_tt='-(gamma_ee + gamma_ee) * i_ee_t - (gamma_ee * gamma_ee) * i_ee + (C_e * T_ee) * gamma_ee * exp(1) * (N_beta_ee * s_e(h_e) + phi_ee + p_ee)'
    i_ei_tt='-(gamma_ei + gamma_ei) * i_ei_t - (gamma_ei * gamma_ei) * i_ei + (C_e * T_ei) * gamma_ei * exp(1) * (N_beta_ei * s_e(h_e) + phi_ei + p_ei)'
    i_ie_tt='-(_gamma(gamma_ie, e_ie) + _gamma_tilde(gamma_ie, e_ie)) * i_ie_t - (_gamma(gamma_ie, e_ie) * _gamma_tilde(gamma_ie, e_ie)) * i_ie + (C_i * T_ie) * _gamma_tilde(gamma_ie, e_ie) * exp(_gamma(gamma_ie, e_ie) / gamma_ie) * (N_beta_ie * s_i(h_i) + phi_ie + p_ie)'
    i_ii_tt='-(_gamma(gamma_ii, e_ii) + _gamma_tilde(gamma_ii, e_ii)) * i_ii_t - (_gamma(gamma_ii, e_ii) * _gamma_tilde(gamma_ii, e_ii)) * i_ii + (C_i * T_ii) * _gamma_tilde(gamma_ii, e_ii) * exp(_gamma(gamma_ii, e_ii) / gamma_ii) * (N_beta_ii * s_i(h_i) + phi_ii + p_ii)'

    phi_ee_tt = '-2 * v * A_ee * phi_ee_t + (v * v) * (A_ee * A_ee) * (N_alpha_ee * s_e(h_e) - phi_ee) + 1.5 * v * v * fake_laplacian'
    phi_ei_tt = '-2 * v * A_ei * phi_ei_t + (v * v) * (A_ei * A_ei) * (N_alpha_ei * s_e(h_e) - phi_ei) + 1.5 * v * v * fake_laplacian'

    zeroIcs = { 'phi_ee' : 0, 'phi_ee_t' : 0, 'phi_ei' : 0, 'phi_ei_t' : 0, 'i_ee' : 0,
                'i_ee_t' : 0, 'i_ei' : 0, 'i_ei_t' : 0, 'i_ie' : 0, 'i_ie_t' : 0,
                'i_ii' : 0, 'i_ii_t' : 0, 'h_e' : 0, 'h_i' : 0,
                'C_i' : 1, 'C_e' : 1}

    def __init__(self, params, ics = None, name="SIRU3", equations = None, points = None, odeSystem = None, timescale = "s"):
        if equations == None:
            print "No eqns for " + name
            equations = { 'phi_ee' : 'phi_ee_t',
                          'phi_ei' : 'phi_ei_t',
                          'phi_ei_t' : SIRU3.phi_ei_tt,
                          'phi_ee_t' : SIRU3.phi_ee_tt,
                          'i_ee' : 'i_ee_t',
                          'i_ei' : 'i_ei_t',
                          'i_ie' : 'i_ie_t',
                          'i_ii' : 'i_ii_t',
                          'i_ee_t' : SIRU3.i_ee_tt,
                          'i_ei_t' : SIRU3.i_ei_tt,
                          'i_ie_t' : SIRU3.i_ie_tt,
                          'i_ii_t' : SIRU3.i_ii_tt,
                          'h_e' : LileyBase.h_e_t,
                          'h_i' : LileyBase.h_i_t,
                          'C_e' : SIRU3.C_e_t,
                          'C_i' : SIRU3.C_i_t}
        else:
            equations = equations

        if ics == None:
            ics = SIRU3.zeroIcs
            ics['h_e'] = params['h_e_rest']
            ics['h_i'] = params['h_i_rest']
            print "ICS.."
            print ics

        LileyBase.__init__(self, params = params, ics = ics, name = name, equations = equations, points = points, odeSystem = odeSystem, timescale = timescale)


class Continuation:
    def __init__(self, odeSystem, cont, sol, name, displayVar, freeVar, point = None):
        self.odeSystem = odeSystem
        self.cont = cont
        self.sol = sol
        self.displayVar = displayVar
        self.name = name
        self.freeVar = freeVar
        self.point = point

    def display(self, displayVar = None, additionalVar = None, fig = "1"):
        if displayVar == None:
            displayVar = self.displayVar
        if additionalVar == None:
            coords = (self.freeVar, displayVar)
        else:
            coords = (self.freeVar, displayVar, additionalVar)

        self.cont[self.name].display(coords, stability = True, figure = fig)
        return self

    def displayMinMax3D(self, x, y, z, fig = "7"):
        figr = figure(fig)
        axes = Axes3D(figr)
        axes.plot(self.cont[self.name].sol[x], self.cont[self.name].sol[y + "_min"], self.cont[self.name].sol[z + "_min"], label='3D')
        axes.plot(self.cont[self.name].sol[x], self.cont[self.name].sol[y + "_max"], self.cont[self.name].sol[z + "_max"], label='3D')
        axes.legend()
        axes.set_xlabel(x)
        axes.set_ylabel(y)
        axes.set_zlabel(z)
        draw()
        return self


    def displayMinMax(self, displayVar = None, fig = "1"):
        if displayVar == None:
            displayVar = self.displayVar

        figure(fig)
        self.cont[self.name].display(coords = (self.freeVar, displayVar), stability = True)
        self.cont[self.name].display(coords = (self.freeVar, displayVar + "_max"), stability = True)
        self.cont[self.name].display(coords = (self.freeVar, displayVar + "_min"), stability = True)
        return self


    def followLimitCycles(self, point, steps = 500, maxStepSize = 1e-3, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        if self.cont[self.name].getSpecialPoint(point) == None:
            print "No point " + point + " for " + self.name
            return None

        newName = self.name + "_cont_" + point
        fullPointName = self.name + ':' + point
        PCargs = args(name=newName, type='LC-C')

        PCargs.initpoint = fullPointName

        PCargs.StepSize = 1e-3 * dirMod
        PCargs.MaxStepSize = maxStepSize
        PCargs.MinStepSize = 1e-8
        PCargs.LocBifPoints = 'all'
        PCargs.FuncTol = 1e-6
        PCargs.VarTol = 1e-6
        PCargs.SolutionMeasures = 'all'
        PCargs.MaxNumPoints = steps
        PCargs.SaveJacobian = True
        PCargs.SaveEigen = True
        PCargs.NumSPOut = steps
        PCargs.freepars = [self.freeVar]
        PCargs.NumCollocation = 6
        PCargs.StopAtPoints = 'B'


        self.cont.newCurve(PCargs)

        self.cont[newName].forward()

        return Continuation(odeSystem = self.odeSystem,
                            cont = self.cont,
                            sol = self.cont[newName].sol,
                            name = newName,
                            displayVar = self.displayVar,
                            freeVar = self.freeVar,
                            point = fullPointName)

    def followLP(self, point, additionalFreeVar, steps = 500, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        newName = self.name + "_cont_" + point
        fullPointName = self.name + ':' + point
        print "Following " + fullPointName
        PCargs = args(name=newName, type='LP-C')

        PCargs.initpoint = fullPointName

        PCargs.StepSize = 1e-3 * dirMod
        PCargs.MaxStepSize = 1e-2
        PCargs.MinStepSize = 1e-12
        PCargs.LocBifPoints = 'all'
        PCargs.FuncTol = 1e-6
        PCargs.VarTol = 1e-6
        PCargs.SolutionMeasures = 'all'
        PCargs.MaxNumPoints = steps
        PCargs.SaveJacobian = True
        PCargs.NumSPOut = steps
        PCargs.freepars = [self.freeVar, additionalFreeVar]

        self.cont.newCurve(PCargs)

        self.cont[newName].forward()

        return Continuation(odeSystem = self.odeSystem,
                            cont = self.cont,
                            sol = self.cont[newName].sol,
                            name = newName,
                            displayVar = self.displayVar,
                            freeVar = self.freeVar,
                            point = fullPointName)

    def followHopf(self, point, additionalFreeVar, steps = 500, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        newName = self.name + "_cont_" + point
        fullPointName = self.name + ':' + point
        print "Following " + fullPointName
        PCargs = args(name=newName, type='H-C1')

        PCargs.initpoint = fullPointName

        PCargs.StepSize = 1e-3 * dirMod
        PCargs.MaxStepSize = 1e-2
        PCargs.MinStepSize = 1e-20
        PCargs.LocBifPoints = 'all'
        PCargs.FuncTol = 1e-3
        PCargs.VarTol = 1e-3
        PCargs.SolutionMeasures = 'all'
        PCargs.MaxNumPoints = steps
        PCargs.SaveJacobian = True
        PCargs.NumSPOut = steps
        PCargs.freepars = [self.freeVar, additionalFreeVar]

        self.cont.newCurve(PCargs)

        self.cont[newName].forward()

        return Continuation(odeSystem = self.odeSystem,
                            cont = self.cont,
                            sol = self.cont[newName].sol,
                            name = newName,
                            displayVar = self.displayVar,
                            freeVar = self.freeVar,
                            point = fullPointName)

    def followHopf2(self, point, additionalFreeVar, steps = 500, maxStepSize = 1e-2, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        newName = self.name + "_cont_" + point + "_ch2_" + additionalFreeVar
        fullPointName = self.name + ':' + point
        PCargs = args(name=newName, type='H-C2')

        PCargs.initpoint = fullPointName

        PCargs.StepSize = 1e-3 * dirMod
        PCargs.MaxStepSize = maxStepSize
        PCargs.MinStepSize = 1e-8
        PCargs.LocBifPoints = ['GH','BT','ZH']
        PCargs.FuncTol = 1e-6
        PCargs.VarTol = 1e-6
        PCargs.SolutionMeasures = 'all'
        PCargs.MaxNumPoints = steps
        PCargs.SaveJacobian = True
        PCargs.NumSPOut = steps
        PCargs.freepars = [self.freeVar, additionalFreeVar]

        self.cont.newCurve(PCargs)

        self.cont[newName].forward()
        self.cont[newName].backward()

        return Continuation(odeSystem = self.odeSystem,
                            cont = self.cont,
                            sol = self.cont[newName].sol,
                            name = newName,
                            displayVar = self.displayVar,
                            freeVar = self.freeVar,
                            point = fullPointName)

    def followSaddleNode(self, point, steps = 500, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        newName = self.name + "_cont_" + point
        fullPointName = self.name + ':' + point
        PCargs = args(name=newName, type='LP-C')

        PCargs.initpoint = fullPointName

        PCargs.StepSize = 1e-3 * dirMod
        PCargs.MaxStepSize = 1e-2
        PCargs.LocBifPoints = 'all'
        PCargs.FuncTol = 1e-6
        PCargs.VarTol = 1e-6
        PCargs.SolutionMeasures = 'all'
        PCargs.MaxNumPoints = steps
        PCargs.SaveJacobian = True
        PCargs.NumSPOut = steps
        PCargs.freepars = [self.freeVar]

        self.cont.newCurve(PCargs)

        self.cont[newName].forward()
        return Continuation(odeSystem = self.odeSystem,
                            cont = self.cont,
                            sol = self.cont[newName].sol,
                            name = newName,
                            displayVar = self.displayVar,
                            freeVar = self.freeVar,
                            point = fullPointName)


    def showCycles(self, coords, fig = "2"):
        figure(fig)
        self.cont[self.name].plot_cycles(coords=coords, figure = fig, method = 'highlight')
        return self

    def vals(self, val):
        return self.cont[self.name].sol[val]

    def specialPoints(self, label):
        return self.cont[self.name].getSpecialPoint(label)

    def specialPointNames(self, prefix = ""):
        return [v['name'] for v in self.cont[self.name].sol.labels[prefix].values()]

    def specialPointsByType(self, prefix = ""):
        return [self.specialPoints(name) for name in self.specialPointNames()]


    def showAll(self):
        show()
        return self












