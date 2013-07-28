/*
 * SIRU2HardcodedParams.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef SIRU2HARDCODEDPARAMS_CUH_
#define SIRU2HARDCODEDPARAMS_CUH_

#include "Params.cuh"
#include "HomogeneousParameterMesh.cuh"
#include "../liley/SIRU2Model.cuh"

class SIRU2HardcodedParams : public Params<SIRU2Model>
{
	ParameterMesh<SIRU2Model> * params(int meshSize)
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

		params[SIRU2Model::T_ee] = 0.3127;
		params[SIRU2Model::T_ei] = 0.9426;
		params[SIRU2Model::T_ie] = 0.4947;
		params[SIRU2Model::T_ii] = 1.4122;

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

//		params[SIRU2Model::p_ee] = 3.6032;
		params[SIRU2Model::p_ee] = 3.61;
		params[SIRU2Model::p_ei] = 0.3639;
		params[SIRU2Model::p_ie] = 0;
		params[SIRU2Model::p_ii] = 0;

		params[SIRU2Model::phi_ie] = 0;
		params[SIRU2Model::phi_ii] = 0;

		params[SIRU2Model::A_ee] = 0.1;
		params[SIRU2Model::A_ei] = 0.1;
//		params[SIRU2Model::A_ee] = 0.2433;
//		params[SIRU2Model::A_ei] = 0.2433;

//		params[SIRU2Model::v] = 0.1714;
		params[SIRU2Model::v] = 0.1;

		params[SIRU2Model::r_abs] = 0;

		params[SIRU2Model::k_i] = 0.1;
		params[SIRU2Model::k_e] = 0.2;

		params[SIRU2Model::e_ee] = 0;
		params[SIRU2Model::e_ei] = 0;
		params[SIRU2Model::e_ie] = 1.5;
		params[SIRU2Model::e_ii] = 1.81;
//		params[SIRU2Model::e_ii] = 1.8;

		return new HomogeneousParameterMesh<SIRU2Model>(params);
	}

	StateSpace initialConditions()
	{
		StateSpace initialConditions(SIRU2Model::NUM_DIMENSIONS);

		initialConditions[SIRU2Model::h_e] = -68.1355;
		initialConditions[SIRU2Model::h_i] = -77.2602;

		initialConditions[SIRU2Model::C_e] = 1;
		initialConditions[SIRU2Model::C_i] = 1;

		return initialConditions;
	}

	map<string, int> stateMap()
	{
		return SIRU2Model::stateMap();
	}

	map<string, int> paramMap()
	{
		return SIRU2Model::paramMap();
	}

};


#endif /* SIRU2HARDCODEDPARAMS_CUH_ */
