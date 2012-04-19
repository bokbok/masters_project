from PyDSTool import *
from pylab import plot, show, figure
from PyDSTool.Toolbox import phaseplane as pp


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

    def __init__(self, params, ics = None, name="LileyBurstBase", equations = None, points = None):
        self.params = params
        self.name = name
        self.points = points
        if equations == None:
            print "No eqns for " + name
            self.equations = { 'phi_ee' : 'phi_ee_t', 'phi_ei' : 'phi_ei_t',
                               'phi_ei_t' : LileyWithBurst.phi_ei_tt, 'phi_ee_t' : LileyWithBurst.phi_ee_tt,
                               'i_ee' : 'i_ee_t', 'i_ei' : 'i_ei_t', 'i_ie' : 'i_ie_t', 'i_ii' : 'i_ii_t',
                               'i_ee_t' : LileyWithBurst.i_ee_tt, 'i_ei_t' : LileyWithBurst.i_ei_tt, 'i_ie_t' : LileyWithBurst.i_ie_tt, 'i_ii_t' : LileyWithBurst.i_ii_tt,
                               'h_e' : LileyWithBurst.h_e_t, 'h_i' : LileyWithBurst.h_i_t, 'slow_e' : LileyWithBurst.slow_e_t, 'slow_i' : LileyWithBurst.slow_i_t }
        else:
            self.equations = equations

        self.auxFunctions = { 'Y_e_h_e' : (['h_e'], LileyWithBurst.Y_e_h_e),
                          'Y_e_h_i' : (['h_i'], LileyWithBurst.Y_e_h_i),
                          'Y_i_h_e' : (['h_e'], LileyWithBurst.Y_i_h_e),
                          'Y_i_h_i' : (['h_i'], LileyWithBurst.Y_i_h_i),
                          's_e' : (['h'], LileyWithBurst.s_e),
                          's_i' : (['h'], LileyWithBurst.s_i)}

        if ics == None:
            self.ics = LileyWithBurst.zeroIcs
            self.ics['h_e'] = params['h_e_rest']
            self.ics['h_i'] = params['h_i_rest']
        else:
            self.ics = ics

    def displayNullclines(self, x, y, fig = "4"):
        odes = self._odeSystem([0, 30])
        subdomain = { x : (-10, 10), y : (-10, 10) }
        for k, v in self.ics.iteritems():
            subdomain[k] = (-10, 10)

        #fp_coord = pp.find_fixedpoints(odes, subdomain=None, n=len(subdomain), eps=1e-8, maxsearch = 30000)
        #print fp_coord
        pp.find_nullclines(odes, x, y, n=3, eps=1e-8,
                                      max_step=0.2, subdomain = subdomain)
    def run(self, timeRange):
        print "Running....", self.params
        self.traj = self._odeSystem(timeRange).compute('run')
        points = self.traj.sample()
        print "Done."

        #set up ICs based on end equilibrium of above run

        contIcs = {}

        for k, v in points.iteritems():
         print k, v
         contIcs[k] = v[len(v) - 1]


        print contIcs

        return LileyWithBurst(params = self.params, ics = contIcs, name = self.name, points = points, equations = self.equations)

    def _odeSystem(self, timeRange):
        self.DSargs = args(varspecs = self.equations, fnspecs = self.auxFunctions, name=self.name)
        self.DSargs.tdomain = timeRange
        self.DSargs.algparams = {'init_step':1e-4, 'atol': 1e-12, 'rtol': 1e-13, 'max_pts' : 10000000}
        self.DSargs.checklevel = 2
        self.DSargs.tdata=timeRange
        self.DSargs.pars = self.params
        self.DSargs.ics = self.ics

        return Radau_ODEsystem(self.DSargs)


    def freeze(self, vars):
        params = self.params.copy()
        ics = self.ics.copy()
        equations = self.equations.copy()

        for var in vars:
            del equations[var]
            params[var] = self.ics[var]
            del ics[var]
        name = self.name + "_freeze_" + str(abs(hash("-".join(vars))))
        print "Freeze ", equations.keys()
        return LileyWithBurst(params = params, ics = ics, name = name, equations = equations)

    def searchForBifurcations(self, freeVar, displayVar, steps = 1000, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1
        odeArgs = args(varspecs = self.equations, fnspecs = self.auxFunctions, name=self.name)
        odeArgs.tdomain = [0, 30]
        odeArgs.algparams = {'init_step':0.00001, 'atol': 1e-12, 'rtol': 1e-13, 'max_pts' : 10000000}
        odeArgs.checklevel = 2
        odeArgs.tdata=[0, 30]
        odeArgs.pars = self.params
        odeArgs.ics = self.ics

        odeSystem = Radau_ODEsystem(odeArgs)

        cont = ContClass(odeSystem)

        contName = self.name + "_cont"
        PCargs = args(name=contName, type='EP-C')
        PCargs.freepars = [freeVar]
        PCargs.StepSize = 1e-6 * dirMod
        PCargs.MaxNumPoints = steps
        PCargs.MaxStepSize = 1e-3
        PCargs.verbosity = 2
        PCargs.FuncTol = 1e-6
        PCargs.VarTol = 1e-6

        PCargs.LocBifPoints = 'all'

        # Declare a new curve based on the above criteria
        cont.newCurve(PCargs)

        # Do path following in the 'forward' direction. Max points is large enough
        # to ensure we go right around the ellipse (PyCont automatically stops when
        # we return to the initial point - unless MaxNumPoints is reached first.)
        cont[contName].forward()

        sol = cont[contName].sol

        return Continuation(odeSystem = odeSystem,
                            cont = cont,
                            sol = sol,
                            name = contName,
                            displayVar = displayVar,
                            freeVar = freeVar)


    def display(self, vars, fig = "1"):
        if self.points == None:
            raise Error("Not run")

        for var in vars:
            figure(fig)
            plot(self.points['t'], self.points[var], label=var)
        return self

class Continuation:

    def __init__(self, odeSystem, cont, sol, name, displayVar, freeVar, point = None):
        self.odeSystem = odeSystem
        self.cont = cont
        self.sol = sol
        self.displayVar = displayVar
        self.name = name
        self.freeVar = freeVar
        self.point = point

    def display(self, fig = "1"):
        figure(fig)
        self.cont.display((self.freeVar, self.displayVar), stability = True)
        return self


    def follow(self, point, steps = 500, dir = '+'):
        if dir == '-':
            dirMod = -1
        else:
            dirMod = 1

        newName = self.name + "_cont_" + point
        fullPointName = self.name + ':' + point
        PCargs = args(name=newName, type='LC-C')

        print fullPointName
        print newName
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
        self.cont[self.name].plot_cycles(coords=coords, figure = fig)
        return self

    def showAll(self):
        show()
        return self












