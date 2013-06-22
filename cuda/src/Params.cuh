/*
 * params.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef PARAMS_CUH_
#define PARAMS_CUH_

#include "ParameterSpace.cuh"
#include "StateSpace.cuh"
#include "liley/SIRU1Model.cuh"
#include "liley/SIRU2Model.cuh"

class Params
{
public:
	ParameterSpace paramsSIRU2()
	{
		ParameterSpace params;
		params[SIRU2Model::tor_e] = 138.3660;
		params[SIRU2Model::tor_i] = 89.3207;
		params[SIRU2Model::h_e_rest] = -68.1355;
		params[SIRU2Model::h_i_rest] = -77.2602;

		params[SIRU2Model::h_ee_eq] = -15.8527;
		params[SIRU2Model::h_ei_eq] = 7.4228;
		params[SIRU2Model::h_ie_eq] = -85.9896;
		params[SIRU2Model::h_ii_eq] = -84.5363;

		params[SIRU2Model::gamma_ee] = 0.4393;
		params[SIRU2Model::gamma_ei] = 0.2350;
		params[SIRU2Model::gamma_ie] = 0.0791;
		params[SIRU2Model::gamma_ii] = 0.0782;

		params[SIRU2Model::T_ee_p] = 0.3127;
		params[SIRU2Model::T_ei_p] = 0.9426;
		params[SIRU2Model::T_ie_p] = 0.4947;
		params[SIRU2Model::T_ii_p] = 1.4122;

		params[SIRU2Model::theta_e] = -46;
		params[SIRU2Model::theta_i] = -46;


		params[SIRU2Model::N_beta_ee] = 4582.0661;
		params[SIRU2Model::N_beta_ei] = 4198.1829;
		params[SIRU2Model::N_beta_ie] = 989.5281;
		params[SIRU2Model::N_beta_ii] = 531.9419;

		params[SIRU2Model::N_alpha_ee] = 4994.4860;
		params[SIRU2Model::N_alpha_ei] = 2222.9060;

		params[SIRU2Model::s_e_max] = 0.2801;
		params[SIRU2Model::s_i_max] = 0.1228;

		params[SIRU2Model::mu_e] = -47.1364;
		params[SIRU2Model::mu_i] = -45.3751;

		params[SIRU2Model::mus_e] = 0.001;
		params[SIRU2Model::mus_i] = 0.001;

		params[SIRU2Model::sigma_e] = 2.6120;
		params[SIRU2Model::sigma_i] = 2.8294;

		params[SIRU2Model::p_ee] = 3.6032;
		params[SIRU2Model::p_ei] = 0.3639;
		params[SIRU2Model::p_ie] = 0;
		params[SIRU2Model::p_ii] = 0;

		params[SIRU2Model::phi_ie] = 0;
		params[SIRU2Model::phi_ii] = 0;

		params[SIRU2Model::A_ee] = 0.2433;
		params[SIRU2Model::A_ei] = 0.2433;

		params[SIRU2Model::v] = 0.1714;

		params[SIRU2Model::r_abs] = 0;

		params[SIRU2Model::k_i] = 0.3;
		params[SIRU2Model::k_e] = 0.3;
		params[SIRU2Model::E_i] = 1e-6;
		params[SIRU2Model::E_e] = 1e-6;

		return params;
	}

	ParameterSpace paramsSIRU1()
	{
		ParameterSpace params;
		params[SIRU1Model::tor_e] = 138.3660;
		params[SIRU1Model::tor_i] = 89.3207;
		params[SIRU1Model::h_e_rest] = -68.1355;
		params[SIRU1Model::h_i_rest] = -77.2602;

		params[SIRU1Model::h_ee_eq] = -15.8527;
		params[SIRU1Model::h_ei_eq] = 7.4228;
		params[SIRU1Model::h_ie_eq] = -85.9896;
		params[SIRU1Model::h_ii_eq] = -84.5363;

		params[SIRU1Model::gamma_ee] = 0.4393;
		params[SIRU1Model::gamma_ei] = 0.2350;
		params[SIRU1Model::gamma_ie] = 0.0791;
		params[SIRU1Model::gamma_ii] = 0.0782;

		params[SIRU1Model::theta_e] = 0.1818;
		params[SIRU1Model::theta_i] = 0.1818;

		params[SIRU1Model::N_beta_ee] = 4582.0661;
		params[SIRU1Model::N_beta_ei] = 4198.1829;
		params[SIRU1Model::N_beta_ie] = 989.5281;
		params[SIRU1Model::N_beta_ii] = 531.9419;

		params[SIRU1Model::N_alpha_ee] = 4994.4860;
		params[SIRU1Model::N_alpha_ei] = 2222.9060;

		params[SIRU1Model::s_e_max] = 0.2801;
		params[SIRU1Model::s_i_max] = 0.1228;

		params[SIRU1Model::mu_e] = -47.1364;
		params[SIRU1Model::mu_i] = -45.3751;

		params[SIRU1Model::mus_ee] = 0.001;
		params[SIRU1Model::mus_ei] = 0.001;
		params[SIRU1Model::mus_ie] = 0.001;
		params[SIRU1Model::mus_ii] = 0.001;


		params[SIRU1Model::sigma_e] = 2.6120;
		params[SIRU1Model::sigma_i] = 2.8294;

		params[SIRU1Model::p_ee] = 3.6032;
		params[SIRU1Model::p_ei] = 0.3639;
		params[SIRU1Model::p_ie] = 0;
		params[SIRU1Model::p_ii] = 0;

		params[SIRU1Model::phi_ie] = 0;
		params[SIRU1Model::phi_ii] = 0;

		params[SIRU1Model::A_ee] = 0.2433;
		params[SIRU1Model::A_ei] = 0.2433;

		params[SIRU1Model::v] = 0.1714;

		params[SIRU1Model::r_abs] = 0;

		params[SIRU1Model::k_i] = 10;
		params[SIRU1Model::k_e] = 10;

		return params;
	}



	StateSpace initialConditionsSIRU2()
	{
		StateSpace initialConditions;

		initialConditions[SIRU2Model::h_e] = 0;
		initialConditions[SIRU2Model::h_i] = 0;

		initialConditions[SIRU2Model::T_ee] = 0.3127;
		initialConditions[SIRU2Model::T_ei] = 0.9426;
		initialConditions[SIRU2Model::T_ie] = 0.4947;
		initialConditions[SIRU2Model::T_ii] = 1.4122;

		return initialConditions;
	}

	StateSpace initialConditionsSIRU1()
	{
		StateSpace initialConditions(SIRU1Model::NUM_DIMENSIONS);

		initialConditions[SIRU1Model::h_e] = -68.1355;
		initialConditions[SIRU1Model::h_i] = -77.2602;

		initialConditions[SIRU1Model::T_ee] = 0.3127;
		initialConditions[SIRU1Model::T_ei] = 0.9426;
		initialConditions[SIRU1Model::T_ie] = 0.4947;
		initialConditions[SIRU1Model::T_ii] = 1.4122;

		return initialConditions;
	}
};

class SIRU1Params
{

};


#endif /* PARAMS_CUH_ */
