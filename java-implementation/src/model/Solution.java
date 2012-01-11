package model;

import integration.ODEGroup;
import integration.ResultsOutput;

import java.io.*;

import static java.lang.Math.*;

public class Solution implements ODEGroup, ResultsOutput
{
    private double h_e_rest;
    private double h_i_rest;
    private double gamma_e;
    private double gamma_i;
    private double N_beta_ee;
    private double N_beta_ei;
    private double N_beta_ie;
    private double N_beta_ii;
    private double N_alpha_ee;
    private double N_alpha_ei;
    private double A_ee;
    private double A_ei;
    private double tor_e;
    private double tor_i;
    private double v;
    private double T_e;
    private double T_i;
    private double p_ee;
    private double p_ei;
    private double p_ie;
    private double p_ii;
    private double s_e_max;
    private double s_i_max;
    private double mu_e;
    private double mu_i;
    private double sigma_e;
    private double sigma_i;
    private double h_e_eq;
    private double h_i_eq;

    public static final int NUM_INTEGRATE = 14;
    public static final int X_STEPS = 50;
    private static final int X_MAX = 50;
    private static final double DELTA_X = (double)X_MAX / (double)X_STEPS;
    private double delta_x_2;
    private double gamma_e_2;
    private double gamma_i_2;
    private double v_2;
    private double A_ee_2;
    private double A_ei_2;
    private PrintWriter out;
    private double _xMax;
    private int _xSteps;

    public Solution(double h_e_rest, double h_i_rest, double gamma_e, double gamma_i,
                    double N_beta_ee, double N_beta_ei,
                    double N_beta_ie, double N_beta_ii,
                    double N_alpha_ee, double N_alpha_ei,
                    double A_ee, double A_ei, double tor_e, double tor_i, double v,
                    double T_e, double T_i,
                    double p_ee, double p_ei, double p_ie, double p_ii, double s_e_max, double s_i_max,
                    double mu_e, double mu_i, double sigma_e, double sigma_i, double h_e_eq, double h_i_eq,
                    double xMax, int xSteps,
                    OutputStream output) throws IOException
    {

        this.h_e_rest = h_e_rest;
        this.h_i_rest = h_i_rest;
        this.gamma_e = gamma_e;
        this.gamma_i = gamma_i;
        this.N_beta_ee = N_beta_ee;
        this.N_beta_ei = N_beta_ei;
        this.N_beta_ie = N_beta_ie;
        this.N_beta_ii = N_beta_ii;
        this.N_alpha_ee = N_alpha_ee;
        this.N_alpha_ei = N_alpha_ei;
        this.A_ee = A_ee;
        this.A_ei = A_ei;
        this.tor_e = tor_e;
        this.tor_i = tor_i;
        this.v = v;
        this.T_e = T_e;
        this.T_i = T_i;
        this.p_ee = p_ee;
        this.p_ei = p_ei;
        this.p_ie = p_ie;
        this.p_ii = p_ii;
        this.s_e_max = s_e_max;
        this.s_i_max = s_i_max;
        this.mu_e = mu_e;
        this.mu_i = mu_i;
        this.sigma_e = sigma_e;
        this.sigma_i = sigma_i;
        this.h_e_eq = h_e_eq;
        this.h_i_eq = h_i_eq;

        this.delta_x_2 = DELTA_X * DELTA_X;
        this.gamma_e_2 = gamma_e * gamma_e;
        this.gamma_i_2 = gamma_i * gamma_i;
        this.v_2 = v * v;
        this.A_ee_2 = A_ee * A_ee;
        this.A_ei_2 = A_ei * A_ei;
        this._xMax = xMax;
        this._xSteps = xSteps;

        this.out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(output)));
    }

    public double[] calculateStep(double[] state, double t)
    {
        double[] derivatives = new double[X_STEPS * NUM_INTEGRATE];

        for(int xStep = 0; xStep < X_STEPS; xStep++)
        {
            XGroup x = new XGroup(state, xStep);

            double phi_e_xx = calcPhi_e_xx(state, xStep);
            double phi_i_xx = calcPhi_i_xx(state, xStep);

            double p_t = (-2 * gamma_e * x.p) - (gamma_e_2 * x.i_ee) + (T_e * gamma_e * exp(1) * ((N_beta_ee * s_e(x.h_e)) + x.phi_e + p_ee));
            double q_t = (-2 * gamma_e * x.q) - (gamma_e_2 * x.i_ei) + (T_e * gamma_e * exp(1) * ((N_beta_ei * s_e(x.h_e)) + x.phi_i + p_ei));

            double r_t = (-2 * gamma_i * x.r) - (gamma_i_2 * x.i_ie) + (T_i * gamma_i * exp(1) * ((N_beta_ie * s_i(x.h_i)) + p_ie));
            double s_t = (-2 * gamma_i * x.s) - (gamma_i_2 * x.i_ii) + (T_i * gamma_i * exp(1) * ((N_beta_ii * s_i(x.h_i)) + p_ii));


            double h_e_t = (1/tor_e) * ((h_e_rest - x.h_e) + (Y_e(x.h_e) * x.i_ee) + (Y_i(x.h_e) * x.i_ie));
            double h_i_t = (1/tor_i) * ((h_i_rest - x.h_i) + (Y_e(x.h_i) * x.i_ei) + (Y_i(x.h_i) * x.i_ii));

            double z_e_t = (-2 * v * A_ee * x.z_e) - (v_2 * A_ee_2 * x.phi_e) + (v_2 * phi_e_xx) + (v_2 * A_ee_2 * N_alpha_ee * s_e(x.h_e));
            double z_i_t = (-2 * v * A_ei * x.z_i) - (v_2 * A_ei_2 * x.phi_i) + (v_2 * phi_i_xx) + (v_2 * A_ei_2 * N_alpha_ei * s_e(x.h_e));

            append_d_dt(derivatives, xStep, p_t, q_t, x.p, x.q, r_t, s_t, x.r, x.s, h_e_t, h_i_t, z_e_t, z_i_t, x.z_e, x.z_i);
        }
        return derivatives;
    }

    private void append_d_dt(double[] derivatives, int xStep, double ... vals)
    {
        int base = xStep * NUM_INTEGRATE;
        for(int i = 0; i < vals.length; i++)
        {
            derivatives[base + i] = vals[i];
        }
    }

    private double s_e(double h_e)
    {
        return s(s_e_max, h_e, mu_e, sigma_e);
    }

    private double s_i(double h_i)
    {
        return s(s_i_max, h_i, mu_i, sigma_i);
    }

    private double s(double max, double h, double mu, double sigma)
    {
        return max / (1 + exp(- sqrt(2) * (h - mu) / sigma));
    }

    private double Y_e(double h_e)
    {
        return Y(h_e, h_e_eq, h_i_rest);
    }

    private double Y_i(double h_i)
    {
        return Y(h_i, h_i_eq, h_e_rest);
    }

    private double Y(double h, double h_eq, double h_rest)
    {
        return (h_eq - h) / abs(h_eq - h_rest);
    }

    private double calcPhi_e_xx(double[] vals, int x_step)
    {
        return calcPhi_xx(vals, x_step, 12);
    }

    private double calcPhi_i_xx(double[] vals, int x_step)
    {
        return calcPhi_xx(vals, x_step, 13);
    }

    private double calcPhi_xx(double[] vals, int x_step, int phi_index)
    {
        double x_minus = extract_x_val(x_step - 1, vals, phi_index);
        double x_plus = extract_x_val(x_step + 1, vals, phi_index);
        double x = extract_x_val(x_step, vals, phi_index);
        return (x_minus - 2 * x + x_plus) / delta_x_2;
    }

    private double extract_x_val(int x_step, double[] vals, int index)
    {
        if (x_step < 0 || x_step == X_STEPS)
        {
            return 0;
        }
        else
        {
            return vals[x_step * NUM_INTEGRATE + index];
        }
    }

    public void write(double t, double[] currentState) throws IOException
    {
        StringBuilder line = new StringBuilder();
        line.append(t);
        for (double val : currentState)
        {
            line.append(':');
            line.append(val);
        }
        out.println(line);
    }

    public void complete()
    {
        out.flush();
        out.close();
    }

    private class XGroup
    {
        private double p;
        private double q;
        private double i_ee;
        private double i_ei;
        private double r;
        private double s;
        private double i_ie;
        private double i_ii;
        private double h_e;
        private double h_i;
        private double z_e;
        private double z_i;
        private double phi_e;
        private double phi_i;

        private XGroup(double[] state, int xStep)
        {
            int base = xStep * NUM_INTEGRATE;
            int index = 0;

            p = state[base + index++];
            q = state[base + index++];
            i_ee = state[base + index++];
            i_ei = state[base + index++];
            r = state[base + index++];
            s = state[base + index++];
            i_ie = state[base + index++];
            i_ii = state[base + index++];
            h_e = state[base + index++];
            h_i = state[base + index++];
            z_e = state[base + index++];
            z_i = state[base + index++];
            phi_e = state[base + index++];
            phi_i = state[base + index++];
        }
    }

    public double[] initialConditions()
    {
        double[] ic = new double[X_STEPS * NUM_INTEGRATE];

        for (int step = 0; step < X_STEPS; step++)
        {
            ic[8 + NUM_INTEGRATE * step] = h_e_rest;
            ic[9 + NUM_INTEGRATE * step] = h_i_rest;
        }
        return ic;
    }


}
