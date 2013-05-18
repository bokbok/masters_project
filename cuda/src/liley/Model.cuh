/*
 * Model.cuh
 *
 *  Created on: 03/05/2013
 *      Author: matt
 */

#ifndef MODEL_CUH_
#define MODEL_CUH_

#include "../Derivatives.cuh"
#include "../StateSpace.cuh"
#include "../DeviceMeshPoint.cuh"

class Model
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
		phi_ei_t
	};

	enum Params
	{
		tor_e,
		tor_i,
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
	Model()
	{
		_e = exp(1.0);
		_sqrt2 = sqrt(2.0);
	}

	__device__
	Derivatives operator ()(StateSpace & state, ParameterSpace & params, DeviceMeshPoint & point)
	{
		Derivatives derivatives;

		derivatives[h_e] = (1/params[tor_e]) * (-(state[h_e] - params[h_e_rest]) + (Y_e_h_e(state[h_e], params) * state[i_ee]) + (Y_i_h_e(state[h_e], params) * (state[i_ie])));
		derivatives[h_i] = (1/params[tor_i]) * (-(state[h_i] - params[h_i_rest]) + (Y_e_h_i(state[h_i], params) * state[i_ei]) + (Y_i_h_i(state[h_i], params) * (state[i_ii])));

		derivatives[i_ee] = state[i_ee_t];
		derivatives[i_ei] = state[i_ei_t];
		derivatives[i_ie] = state[i_ie_t];
		derivatives[i_ii] = state[i_ii_t];

	    derivatives[i_ee_t] = -2 * params[gamma_ee] * state[i_ee_t] - (params[gamma_ee] * params[gamma_ee]) * state[i_ee] + params[T_ee] * params[gamma_ee] * _e * (params[N_beta_ee] * s_e(state[h_e], params) + state[phi_ee] + params[p_ee]);
	    derivatives[i_ei_t] = -2 * params[gamma_ei] * state[i_ei_t] - (params[gamma_ei] * params[gamma_ei]) * state[i_ei] + params[T_ei] * params[gamma_ei] * _e * (params[N_beta_ei] * s_e(state[h_e], params) + state[phi_ei] + params[p_ei]);

	    derivatives[i_ie_t] = -2 * params[gamma_ie] * state[i_ie_t] - (params[gamma_ie] * params[gamma_ie]) * state[i_ie] + params[T_ie] * params[gamma_ie] * _e * (params[N_beta_ie] * s_i(state[h_i], params) + params[phi_ie] + params[p_ie]);
	    derivatives[i_ii_t] = -2 * params[gamma_ii] * state[i_ii_t] - (params[gamma_ii] * params[gamma_ii]) * state[i_ii] + params[T_ii] * params[gamma_ii] * _e * (params[N_beta_ii] * s_i(state[h_i], params) + params[phi_ii] + params[p_ii]);

	    derivatives[phi_ee] = state[phi_ee_t];
	    derivatives[phi_ee] = state[phi_ee_t];

	    double vel = params[v];
	    double vel_squared = vel * vel;

	    derivatives[phi_ee_t] = -2 * vel * params[A_ee] * state[phi_ee_t] + vel_squared * (params[A_ee] * params[A_ee]) * (params[N_alpha_ee] * s_e(state[h_e], params) - state[phi_ee]);
	    derivatives[phi_ei_t] = -2 * vel * params[A_ei] * state[phi_ei_t] + vel_squared * (params[A_ei] * params[A_ei]) * (params[N_alpha_ei] * s_e(state[h_e], params) - state[phi_ei]);
		return derivatives;
	}

	__device__
	double Y_e_h_e(double h_e, ParameterSpace & params)
	{
		return (params[h_ee_eq] - h_e) / abs(params[h_ee_eq] - params[h_e_rest]);
	}

	__device__
	double Y_e_h_i(double h_i, ParameterSpace & params)
	{
		return (params[h_ei_eq] - h_i) / abs(params[h_ei_eq] - params[h_i_rest]);
	}

	__device__
	double Y_i_h_e(double h_e, ParameterSpace & params)
	{
		return (params[h_ie_eq] - h_e) / abs(params[h_ie_eq] - params[h_e_rest]);
	}

	__device__
	double Y_i_h_i(double h_i, ParameterSpace & params)
	{
		return (params[h_ii_eq] - h_i) / abs(params[h_ii_eq] - params[h_i_rest]);
	}

	__device__
    double s_e(double h, ParameterSpace & params)
    {
    	return params[s_e_max] / (1 + (1 - params[r_abs] * params[s_e_max]) * exp(-_sqrt2 * (h - params[mu_e]) / params[sigma_e]));
    }

	__device__
    double s_i(double h, ParameterSpace & params)
    {
    	return params[s_i_max] / (1 + (1 - params[r_abs] * params[s_i_max]) * exp(-_sqrt2 * (h - params[mu_i]) / params[sigma_i]));
    }
};


#endif /* MODEL_CUH_ */
