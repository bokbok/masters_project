package model;

import integration.RungeKutta;

import java.io.*;
import java.util.Map;

public class Main
{
    public static void main(String[] args) throws IOException
    {
        //# h_e_rest, h_i_rest, gamma_e, gamma_i,
        //# N_beta_ee, N_beta_ei,
        //# N_beta_ie, N_beta_ii,
        //# N_alpha_ee, N_alpha_ei,
        //# A_ee, A_ei, tor_e, tor_i, v,
        //# T_e, T_i,
        //# p_ee, p_ei, p_ie, p_ii,
        //# s_e_max, s_i_max, mu_e, mu_i,
        //# sigma_e, sigma_i, h_e_eq, h_i_eq

        Solution solution = new Solution(-70, -70, 300, 65,
                                         3034, 3034,
                                         536, 536,
                                         4000, 2000,
                                         0.4, 0.4, 0.01, 0.01, 700,
                                         0.4, 0.8,
                                         0, 0, 0, 0,
                                         500, 500, -50, -50,
                                         5, 5, 45, -90);

        RungeKutta integrator = new RungeKutta(20, 20 * 100000);

        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("c:\\temp\\run.dat")));
        integrator.fourthOrder(solution, solution);

        out.flush();
        out.close();

    }
}
