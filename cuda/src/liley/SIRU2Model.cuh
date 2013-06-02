/*
 * SIRU.cuh
 *
 *  Created on: 19/05/2013
 *      Author: matt
 */

#ifndef SIRU2_CUH_
#define SIRU2_CUH_

class SIRU2Model
{
private:
	double _e, _sqrt2;

public:
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
		T_ee,
		T_ei,
		T_ie,
		T_ii,
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
		T_ee_p,
		T_ei_p,
		T_ie_p,
		T_ii_p,
		mus_e,
		mus_i,
		k_i,
		k_e,
		e_i,
		e_e,
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
		r_abs
	};

	__host__ __device__
	SIRU2Model()
	{
		_e = exp(1.0);
		_sqrt2 = sqrt(2.0);
	}

	__device__
	Derivatives operator ()(StateSpace & s, ParameterSpace & p, DeviceMeshPoint & point)
	{
		Derivatives ddt;
	    double vel = p[v];
	    double vel2 = vel * vel;

		ddt[h_e] = (1/p[tor_e]) * (-(s[h_e] - p[h_e_rest]) + (Y_e_h_e(s[h_e], p) * s[i_ee]) + (Y_i_h_e(s[h_e], p) * (s[i_ie])));
		ddt[h_i] = (1/p[tor_i]) * (-(s[h_i] - p[h_i_rest]) + (Y_e_h_i(s[h_i], p) * s[i_ei]) + (Y_i_h_i(s[h_i], p) * (s[i_ii])));

		ddt[i_ee] = s[i_ee_t];
		ddt[i_ei] = s[i_ei_t];
		ddt[i_ie] = s[i_ie_t];
		ddt[i_ii] = s[i_ii_t];

		ddt[T_ii] = p[mus_i] * (p[T_ii_p] / (1 + exp(p[k_i] * (s[h_i] - p[theta_i]))) - s[T_ii]);
		ddt[T_ie] = p[mus_i] * (p[T_ie_p] / (1 + exp(p[k_i] * (s[h_i] - p[theta_i]))) - s[T_ie]);

		ddt[T_ei] = p[mus_e] * (p[T_ei_p] / (1 + exp(p[k_e] * (s[h_e] - p[theta_e]))) - s[T_ei]);
		ddt[T_ee] = p[mus_e] * (p[T_ee_p] / (1 + exp(p[k_e] * (s[h_e] - p[theta_e]))) - s[T_ee]);

	    ddt[i_ee_t] = -(g1(gamma_ee, e_e, p) + g2(gamma_ee, e_e, p)) * s[i_ee_t] - (g1(gamma_ee, e_e, p) * g2(gamma_ee, e_e, p)) * s[i_ee] + s[T_ee] * g2(gamma_ee, e_e, p) * _e * (p[N_beta_ee] * s_e(s[h_e], p) + s[phi_ee] + p[p_ee]);
	    ddt[i_ei_t] = -(g1(gamma_ei, e_e, p) + g2(gamma_ei, e_e, p)) * s[i_ei_t] - (g1(gamma_ei, e_e, p) * g2(gamma_ei, e_e, p)) * s[i_ei] + s[T_ei] * g2(gamma_ei, e_e, p) * _e * (p[N_beta_ei] * s_e(s[h_e], p) + s[phi_ei] + p[p_ei]);

	    ddt[i_ie_t] = -(g1(gamma_ie, e_i, p) + g2(gamma_ie, e_i, p)) * s[i_ie_t] - (g1(gamma_ie, e_i, p) * g2(gamma_ie, e_i, p)) * s[i_ie] + s[T_ie] * g2(gamma_ie, e_i, p) * _e * (p[N_beta_ie] * s_i(s[h_i], p) + p[phi_ie] + p[p_ie]);
	    ddt[i_ii_t] = -(g1(gamma_ii, e_i, p) + g2(gamma_ii, e_i, p)) * s[i_ii_t] - (g1(gamma_ii, e_i, p) * g2(gamma_ii, e_i, p)) * s[i_ii] + s[T_ii] * g2(gamma_ii, e_i, p) * _e * (p[N_beta_ii] * s_i(s[h_i], p) + p[phi_ii] + p[p_ii]);

	    ddt[phi_ee] = s[phi_ee_t];
	    ddt[phi_ei] = s[phi_ei_t];

	    ddt[phi_ee_t] = -2 * vel * p[A_ee] * s[phi_ee_t] + vel2 * (p[A_ee] * p[A_ee]) * (p[N_alpha_ee] * s_e(s[h_e], p) - s[phi_ee]) + 3 * vel2 * (point.laplacian(phi_ee)) / 2;
	    ddt[phi_ei_t] = -2 * vel * p[A_ei] * s[phi_ei_t] + vel2 * (p[A_ei] * p[A_ei]) * (p[N_alpha_ei] * s_e(s[h_e], p) - s[phi_ei]) + 3 * vel2 * (point.laplacian(phi_ei)) / 2;
		return ddt;
	}

	__device__
	double Y_e_h_e(double h_e, ParameterSpace & p)
	{
		return (p[h_ee_eq] - h_e) / abs(p[h_ee_eq] - p[h_e_rest]);
	}

	__device__
	double g1(int param, int rate, ParameterSpace & p)
	{
		double e_lk = p[rate];
		double gamma_lk = p[param];


		return e_lk * gamma_lk / ( exp(e_lk) - 1 );
	}

	__device__
	double g2(int param, int rate, ParameterSpace & p)
	{
		double e_lk = p[rate];
		double gamma_lk = p[param];


		return g1(param, rate, p) * exp(e_lk);
	}

	__device__
	double Y_e_h_i(double h_i, ParameterSpace & p)
	{
		return (p[h_ei_eq] - h_i) / abs(p[h_ei_eq] - p[h_i_rest]);
	}

	__device__
	double Y_i_h_e(double h_e, ParameterSpace & p)
	{
		return (p[h_ie_eq] - h_e) / abs(p[h_ie_eq] - p[h_e_rest]);
	}

	__device__
	double Y_i_h_i(double h_i, ParameterSpace & p)
	{
		return (p[h_ii_eq] - h_i) / abs(p[h_ii_eq] - p[h_i_rest]);
	}

	__device__
    double s_e(double h, ParameterSpace & p)
    {
    	return p[s_e_max] / (1 + (1 - p[r_abs] * p[s_e_max]) * exp(-_sqrt2 * (h - p[mu_e]) / p[sigma_e]));
    }

	__device__
    double s_i(double h, ParameterSpace & p)
    {
    	return p[s_i_max] / (1 + (1 - p[r_abs] * p[s_i_max]) * exp(-_sqrt2 * (h - p[mu_i]) / p[sigma_i]));
    }
};


#endif /* SIRU2_CUH_ */
