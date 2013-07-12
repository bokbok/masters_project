/*
 * SIRU.cuh
 *
 *  Created on: 19/05/2013
 *      Author: matt
 */

#ifndef SIRU2_CUH_
#define SIRU2_CUH_
#include "../ParameterSpace.cuh"
#include "../DeviceMeshPoint.cuh"
#include <string>

using namespace std;

class SIRU2Model
{
private:
	double _1_5_vel2_laplacian_phi_ee,
		   _1_5_vel2_laplacian_phi_ei;

public:
	static const int NUM_DIMENSIONS = 16;
	enum Dimensions
	{
		h_e,
		h_i,
		i_ee,
		i_ei,
		i_ie,
		i_ii,
		i_ee_t,
		i_ei_t,
		i_ie_t,
		i_ii_t,
		phi_ee,
		phi_ei,
		phi_ee_t,
		phi_ei_t,
		C_e,
		C_i
	};

	enum Params
	{
		tor_e,
		tor_i,
		theta_e,
		theta_i,
		h_e_rest,
		h_i_rest,
		h_ee_eq,
		h_ei_eq,
		h_ie_eq,
		h_ii_eq,
		gamma_ee,
		gamma_ei,
		gamma_ie,
		gamma_ii,
		T_ee,
		T_ei,
		T_ie,
		T_ii,
		mus_e,
		mus_i,
		k_i,
		k_e,
		E_i,
		E_e,
		N_beta_ee,
		N_beta_ei,
		N_beta_ie,
		N_beta_ii,
		N_alpha_ee,
		N_alpha_ei,
		s_e_max,
		s_i_max,
		mu_e,
		mu_i,
		sigma_e,
		sigma_i,
		p_ee,
		p_ei,
		p_ie,
		p_ii,
		phi_ie,
		phi_ii,
		v,
		A_ee,
		A_ei,
		r_abs,

		e_ee,
		e_ei,
		e_ie,
		e_ii
	};

	enum Precalculations
	{
		_e = 60,
		_sqrt2,

		_vel,
		_vel2,

		_abs_h_ee_eq_minus_h_e_rest,
		_abs_h_ei_eq_minus_h_i_rest,
		_abs_h_ie_eq_minus_h_e_rest,
		_abs_h_ii_eq_minus_h_i_rest,

		_gamma_ee_tilde,
		_gamma_ei_tilde,
		_gamma_ie_tilde,
		_gamma_ii_tilde,

		_gamma_ee,
		_gamma_ei,
		_gamma_ie,
		_gamma_ii,

		_gamma_ee_tilde2,
		_gamma_ei_tilde2,
		_gamma_ie_tilde2,
		_gamma_ii_tilde2,

		_minus_2_vel_A_ee,
		_minus_2_vel_A_ei,

		_vel2_A_ee2,
		_vel2_A_ei2,

		_T_ee_gamma_ee_e,
		_T_ei_gamma_ei_e,
		_T_ie_gamma_ie_e,
		_T_ii_gamma_ii_e,

		_1_minus_r_abs_s_e_max,
		_1_minus_r_abs_s_i_max,

		_minus_sqrt2_div_sigma_e,
		_minus_sqrt2_div_sigma_i
	};

	__host__
	static map<string, int> stateMap()
	{
		map<string, int> result;
		result["h_e"] = h_e;
		result["h_i"] = h_i;
		result["i_ee"] = i_ee;
		result["i_ei"] = i_ei;
		result["i_ie"] = i_ie;
		result["i_ii"] = i_ii;
		result["i_ee_t"] = i_ee_t;
		result["i_ei_t"] = i_ei_t;
		result["i_ie_t"] = i_ie_t;
		result["i_ii_t"] = i_ii_t;
		result["phi_ee"] = phi_ee;
		result["phi_ei"] = phi_ei;
		result["phi_ee_t"] = phi_ee_t;
		result["phi_ei_t"] = phi_ei_t;
		result["C_e"] = C_e;
		result["C_i"] = C_i;

		return result;
	}

	__host__
	static map<string, int> paramMap()
	{
		map<string, int> result;
		result["tor_e"] = tor_e;
		result["tor_i"] = tor_i;
		result["theta_e"] = theta_e;
		result["theta_i"] = theta_i;
		result["h_e_rest"] = h_e_rest;
		result["h_i_rest"] = h_i_rest;
		result["h_ee_eq"] = h_ee_eq;
		result["h_ei_eq"] = h_ei_eq;
		result["h_ie_eq"] = h_ie_eq;
		result["h_ii_eq"] = h_ii_eq;
		result["gamma_ee"] = gamma_ee;
		result["gamma_ei"] = gamma_ei;
		result["gamma_ie"] = gamma_ie;
		result["gamma_ii"] = gamma_ii;
		result["T_ee"] = T_ee;
		result["T_ei"] = T_ei;
		result["T_ie"] = T_ie;
		result["T_ii"] = T_ii;
		result["mus_e"] = mus_e;
		result["mus_i"] = mus_i;
		result["k_i"] = k_i;
		result["k_e"] = k_e;
		result["E_i"] = E_i;
		result["E_e"] = E_e;
		result["N_beta_ee"] = N_beta_ee;
		result["N_beta_ei"] = N_beta_ei;
		result["N_beta_ie"] = N_beta_ie;
		result["N_beta_ii"] = N_beta_ii;
		result["N_alpha_ee"] = N_alpha_ee;
		result["N_alpha_ei"] = N_alpha_ei;
		result["s_e_max"] = s_e_max;
		result["s_i_max"] = s_i_max;
		result["mu_e"] = mu_e;
		result["mu_i"] = mu_i;
		result["sigma_e"] = sigma_e;
		result["sigma_i"] = sigma_i;
		result["p_ee"] = p_ee;
		result["p_ei"] = p_ei;
		result["p_ie"] = p_ie;
		result["p_ii"] = p_ii;
		result["phi_ie"] = phi_ie;
		result["phi_ii"] = phi_ii;
		result["v"] = v;
		result["A_ee"] = A_ee;
		result["A_ei"] = A_ei;
		result["r_abs"] = r_abs;
		result["e_ee"] = e_ee;
		result["e_ei"] = e_ei;
		result["e_ie"] = e_ie;
		result["e_ii"] = e_ii;

		return result;
	}


	__host__
	static void precalculate(ParameterSpace &p)
	{
		p[_e] = exp(1.0);
		p[_sqrt2] = sqrt(2.0);

		p[_vel] = p[v];
		p[_vel2] = p[_vel] * p[_vel];

		p[_abs_h_ee_eq_minus_h_e_rest] = abs(p[h_ee_eq] - p[h_e_rest]);
		p[_abs_h_ei_eq_minus_h_i_rest] = abs(p[h_ei_eq] - p[h_i_rest]);
		p[_abs_h_ie_eq_minus_h_e_rest] = abs(p[h_ie_eq] - p[h_e_rest]);
		p[_abs_h_ii_eq_minus_h_i_rest] = abs(p[h_ii_eq] - p[h_i_rest]);

		if (p[e_ee] > 0)
		{
			p[_gamma_ee] = p[gamma_ee] * p[e_ee] / (exp(p[e_ee]) - 1);
		}
		else
		{
			p[_gamma_ee] = p[gamma_ee];
		}

		if (p[e_ei] > 0)
		{
			p[_gamma_ei] = p[gamma_ei] * p[e_ei] / (exp(p[e_ei]) - 1);
		}
		else
		{
			p[_gamma_ei] = p[gamma_ei];
		}

		if (p[e_ie] > 0)
		{
			p[_gamma_ie] = p[gamma_ie] * p[e_ie] / (exp(p[e_ie]) - 1);
		}
		else
		{
			p[_gamma_ie] = p[gamma_ie];
		}

		if (p[e_ii] > 0)
		{
			p[_gamma_ii] = p[gamma_ii] * p[e_ii] / (exp(p[e_ii]) - 1);
		}
		else
		{
			p[_gamma_ii] = p[gamma_ii];
		}

		p[_gamma_ee_tilde] = p[_gamma_ee] * exp(p[e_ee]);
		p[_gamma_ei_tilde] = p[_gamma_ei] * exp(p[e_ei]);
		p[_gamma_ie_tilde] = p[_gamma_ie] * exp(p[e_ie]);
		p[_gamma_ii_tilde] = p[_gamma_ii] * exp(p[e_ii]);

		p[_gamma_ee_tilde2] = p[_gamma_ee] * p[_gamma_ee_tilde];
		p[_gamma_ei_tilde2] = p[_gamma_ei] * p[_gamma_ei_tilde];
		p[_gamma_ie_tilde2] = p[_gamma_ie] * p[_gamma_ie_tilde];
		p[_gamma_ii_tilde2] = p[_gamma_ii] * p[_gamma_ii_tilde];

		p[_minus_2_vel_A_ee] = -2 * p[_vel] * p[A_ee];
		p[_minus_2_vel_A_ei] = -2 * p[_vel] * p[A_ei];

		p[_vel2_A_ee2] = p[_vel2] * p[A_ee] * p[A_ee];
		p[_vel2_A_ei2] = p[_vel2] * p[A_ei] * p[A_ei];

		p[_T_ee_gamma_ee_e] = p[T_ee] * p[_gamma_ee_tilde] * exp(p[_gamma_ee] / p[gamma_ee]);
		p[_T_ei_gamma_ei_e] = p[T_ei] * p[_gamma_ei_tilde] * exp(p[_gamma_ei] / p[gamma_ei]);
		p[_T_ie_gamma_ie_e] = p[T_ie] * p[_gamma_ie_tilde] * exp(p[_gamma_ie] / p[gamma_ie]);
		p[_T_ii_gamma_ii_e] = p[T_ii] * p[_gamma_ii_tilde] * exp(p[_gamma_ii] / p[gamma_ii]);

		p[_1_minus_r_abs_s_e_max] = 1 - p[r_abs] * p[s_e_max];
		p[_1_minus_r_abs_s_i_max] = 1 - p[r_abs] * p[s_i_max];

		p[_minus_sqrt2_div_sigma_e] = -p[_sqrt2] / p[sigma_e];
		p[_minus_sqrt2_div_sigma_i] = -p[_sqrt2] / p[sigma_i];
	}

	__host__
	SIRU2Model()
	{

	}

	__device__
	SIRU2Model(ParameterSpace & p, DeviceMeshPoint & pt)
	{
		_1_5_vel2_laplacian_phi_ee = 3 * p[_vel2] * (pt.laplacian(phi_ee)) / 2;
		_1_5_vel2_laplacian_phi_ei = 3 * p[_vel2] * (pt.laplacian(phi_ei)) / 2;
	}
	__device__
	inline void operator ()(double * ddt, double * s, double * p, DeviceMeshPoint & point)
	{
		ddt[h_e] = (-(s[h_e] - p[h_e_rest]) + (Y_e_h_e(s[h_e], p) * s[i_ee]) + (Y_i_h_e(s[h_e], p) * (s[i_ie]))) / p[tor_e];
		ddt[h_i] = (-(s[h_i] - p[h_i_rest]) + (Y_e_h_i(s[h_i], p) * s[i_ei]) + (Y_i_h_i(s[h_i], p) * (s[i_ii]))) / p[tor_i];

		ddt[C_i] = p[mus_i] * ((1 / (1 + exp(p[k_i] * (s[h_i] - p[theta_i]))))  - s[C_i]);
		ddt[C_e] = p[mus_e] * ((1 / (1 + exp(p[k_e] * (s[h_e] - p[theta_e]))))  - s[C_e]);

	    ddt[i_ee_t] = -(p[_gamma_ee] + p[_gamma_ee_tilde]) * s[i_ee_t] - p[_gamma_ee_tilde2] * s[i_ee] + s[C_e] * p[_T_ee_gamma_ee_e] * (p[N_beta_ee] * s_e(s[h_e], p) + s[phi_ee] + p[p_ee]);
	    ddt[i_ei_t] = -(p[_gamma_ei] + p[_gamma_ei_tilde]) * s[i_ei_t] - p[_gamma_ei_tilde2] * s[i_ei] + s[C_e] * p[_T_ei_gamma_ei_e] * (p[N_beta_ei] * s_e(s[h_e], p) + s[phi_ei] + p[p_ei]);

	    ddt[i_ie_t] = -(p[_gamma_ie] + p[_gamma_ie_tilde]) * s[i_ie_t] - p[_gamma_ie_tilde2] * s[i_ie] + s[C_i] * p[_T_ie_gamma_ie_e] * (p[N_beta_ie] * s_i(s[h_i], p) + p[phi_ie] + p[p_ie]);
	    ddt[i_ii_t] = -(p[_gamma_ii] + p[_gamma_ii_tilde]) * s[i_ii_t] - p[_gamma_ii_tilde2] * s[i_ii] + s[C_i] * p[_T_ii_gamma_ii_e] * (p[N_beta_ii] * s_i(s[h_i], p) + p[phi_ii] + p[p_ii]);

	    ddt[phi_ee_t] = p[_minus_2_vel_A_ee] * s[phi_ee_t] + p[_vel2_A_ee2] * (p[N_alpha_ee] * s_e(s[h_e], p) - s[phi_ee]) + _1_5_vel2_laplacian_phi_ee;
	    ddt[phi_ei_t] = p[_minus_2_vel_A_ei] * s[phi_ei_t] + p[_vel2_A_ei2] * (p[N_alpha_ei] * s_e(s[h_e], p) - s[phi_ei]) + _1_5_vel2_laplacian_phi_ei;

	    ddt[phi_ee] = s[phi_ee_t];
	    ddt[phi_ei] = s[phi_ei_t];

	    ddt[i_ee] = s[i_ee_t];
		ddt[i_ei] = s[i_ei_t];
		ddt[i_ie] = s[i_ie_t];
		ddt[i_ii] = s[i_ii_t];
	}

	__device__
	inline double Y_e_h_e(double h_e, double * p)
	{
		return (p[h_ee_eq] - h_e) / p[_abs_h_ee_eq_minus_h_e_rest];
	}

	__device__
	inline double Y_e_h_i(double h_i, double * p)
	{
		return (p[h_ei_eq] - h_i) / p[_abs_h_ei_eq_minus_h_i_rest];
	}

	__device__
	inline double Y_i_h_e(double h_e, double * p)
	{
		return (p[h_ie_eq] - h_e) / p[_abs_h_ie_eq_minus_h_e_rest];
	}

	__device__
	inline double Y_i_h_i(double h_i, double * p)
	{
		return (p[h_ii_eq] - h_i) / p[_abs_h_ii_eq_minus_h_i_rest];
	}

	__device__
    inline double s_e(double h, double * p)
    {
    	return p[s_e_max] / (1 + (p[_1_minus_r_abs_s_e_max]) * exp(p[_minus_sqrt2_div_sigma_e] * (h - p[mu_e])));
    }

	__device__
    inline double s_i(double h, double * p)
    {
    	return p[s_i_max] / (1 + (p[_1_minus_r_abs_s_i_max]) * exp(p[_minus_sqrt2_div_sigma_i] *(h - p[mu_i])));
    }
};


#endif /* SIRU2_CUH_ */
