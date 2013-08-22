/*
 * SIRU3NonHomogeneousParams.cuh
 *
 *  Created on: 25/07/2013
 *      Author: matt
 */

#ifndef SIRU3NONHOMOGENEOUSPARAMS_CUH_
#define SIRU3NONHOMOGENEOUSPARAMS_CUH_

#include "Params.cuh"
#include "../ParameterSpace.cuh"
#include "NonHomogeneousParameterMesh.cuh"
#include "../liley/SIRU3Model.cuh"


class SIRU3NonHomogeneousParams : public Params<SIRU3Model>
{
private:
	ParameterSpace base()
	{
		ParameterSpace params;
		params[SIRU3Model::tor_e] = 138.3660;
		params[SIRU3Model::tor_i] = 89.3207;
		params[SIRU3Model::h_e_rest] = -68.1355;
		params[SIRU3Model::h_i_rest] = -77.2602;

		params[SIRU3Model::h_ee_eq] = -15.8527;
		params[SIRU3Model::h_ei_eq] = 7.4228;
		params[SIRU3Model::h_ie_eq] = -85.9896;
		params[SIRU3Model::h_ii_eq] = -84.5363;

		params[SIRU3Model::gamma_ee] = 0.4393;
		params[SIRU3Model::gamma_ei] = 0.2350;
		params[SIRU3Model::gamma_ie] = 0.0791;
		params[SIRU3Model::gamma_ii] = 0.0782;

		params[SIRU3Model::T_ee] = 0.3127;
		params[SIRU3Model::T_ei] = 0.9426;
		params[SIRU3Model::T_ie] = 0.4947;
		params[SIRU3Model::T_ii] = 1.4122;

		params[SIRU3Model::theta_e] = -46;
		params[SIRU3Model::theta_i] = -46;

		params[SIRU3Model::N_beta_ee] = 4582.0661;
		params[SIRU3Model::N_beta_ei] = 4198.1829;
		params[SIRU3Model::N_beta_ie] = 989.5281;
		params[SIRU3Model::N_beta_ii] = 531.9419;

		params[SIRU3Model::N_alpha_ee] = 4994.4860;
		params[SIRU3Model::N_alpha_ei] = 2222.9060;

		params[SIRU3Model::s_e_max] = 0.2801;
		params[SIRU3Model::s_i_max] = 0.1228;

		params[SIRU3Model::mu_e] = -47.1364;
		params[SIRU3Model::mu_i] = -45.3751;

		params[SIRU3Model::mus_e] = 0.001;
		params[SIRU3Model::mus_i] = 0.001;

		params[SIRU3Model::sigma_e] = 2.6120;
		params[SIRU3Model::sigma_i] = 2.8294;

//		params[SIRU3Model::p_ee] = 3.6032;
		params[SIRU3Model::p_ee] = 3.485;
		params[SIRU3Model::p_ei] = 0.3639;
		params[SIRU3Model::p_ie] = 0;
		params[SIRU3Model::p_ii] = 0;

		params[SIRU3Model::phi_ie] = 0;
		params[SIRU3Model::phi_ii] = 0;

		params[SIRU3Model::A_ee] = 0.2433;
		params[SIRU3Model::A_ei] = 0.2433;

//		params[SIRU3Model::v] = 0.1714;
		params[SIRU3Model::v] = 0.07;

		params[SIRU3Model::r_abs] = 0;

		params[SIRU3Model::g_i] = 0.1;
		params[SIRU3Model::g_e] = 0.2;

		params[SIRU3Model::e_ee] = 0;
		params[SIRU3Model::e_ei] = 0;
		params[SIRU3Model::e_ie] = 1.5;
//		params[SIRU3Model::e_ii] = 1.8;
		params[SIRU3Model::e_ii] = 1.72;

		return params;
	}

	const static int SAMPLE_POINTS = 5;
public:
	ParameterMesh<SIRU3Model> * mesh(int meshSize)
	{
		NonHomogeneousParameterMesh<SIRU3Model> * mesh = new NonHomogeneousParameterMesh<SIRU3Model>(SAMPLE_POINTS, meshSize);

		ParameterSpace point1 = base();
		point1[SIRU3Model::v] = 0.185;
		point1[SIRU3Model::p_ee] = 3;
		point1[SIRU3Model::e_ii] = 2.5;
		point1[SIRU3Model::mus_e] = 0;
		point1[SIRU3Model::mus_i] = 0;


		ParameterSpace point2 = base();
		point2[SIRU3Model::v] = 0.185;
		point2[SIRU3Model::p_ee] = 3;
		point2[SIRU3Model::e_ii] = 2.5;
		point2[SIRU3Model::mus_e] = 0;
		point2[SIRU3Model::mus_i] = 0;

		ParameterSpace point3 = base();
		point3[SIRU3Model::v] = 0.185;
		point3[SIRU3Model::p_ee] = 3;
		point3[SIRU3Model::e_ii] = 2.5;
		point3[SIRU3Model::mus_e] = 0;
		point3[SIRU3Model::mus_i] = 0;

		ParameterSpace point4 = base();
		point4[SIRU3Model::v] = 0.185;
		point4[SIRU3Model::p_ee] = 3;
		point4[SIRU3Model::e_ii] = 2.5;
		point4[SIRU3Model::mus_e] = 0;
		point4[SIRU3Model::mus_i] = 0;

		ParameterSpace point5 = base();
		point5[SIRU3Model::v] = 0.185;
		point5[SIRU3Model::p_ee] = 3;
		point5[SIRU3Model::e_ii] = 2.5;
		point5[SIRU3Model::mus_e] = 0;
		point5[SIRU3Model::mus_i] = 0;

		ParameterSpace point6 = base();
		point6[SIRU3Model::v] = 0.185;
		point6[SIRU3Model::p_ee] = 3;
		point6[SIRU3Model::e_ii] = 2.5;
		point6[SIRU3Model::mus_e] = 0;
		point6[SIRU3Model::mus_i] = 0;

		ParameterSpace point7 = base();
		point7[SIRU3Model::v] = 0.185;
		point7[SIRU3Model::p_ee] = 3;
		point7[SIRU3Model::e_ii] = 2.5;
		point7[SIRU3Model::mus_e] = 0;
		point7[SIRU3Model::mus_i] = 0;

		ParameterSpace point8 = base();
		point5[SIRU3Model::v] = 0.185;
		point5[SIRU3Model::p_ee] = 3;
		point5[SIRU3Model::e_ii] = 2.5;
		point5[SIRU3Model::mus_e] = 0;
		point5[SIRU3Model::mus_i] = 0;


		mesh->addRefPoint(0, 0, point1);
		mesh->addRefPoint(0, 1, point2);
		mesh->addRefPoint(0, 2, point6);
		mesh->addRefPoint(0, 3, point2);
		mesh->addRefPoint(0, 4, point1);

		mesh->addRefPoint(1, 0, point3);
		mesh->addRefPoint(1, 1, point4);
		mesh->addRefPoint(1, 2, point4);
		mesh->addRefPoint(1, 3, point4);
		mesh->addRefPoint(1, 4, point3);

		mesh->addRefPoint(2, 0, point5);
		mesh->addRefPoint(2, 1, point7);
		mesh->addRefPoint(2, 2, point8); // centre
		mesh->addRefPoint(2, 3, point7);
		mesh->addRefPoint(2, 4, point5);

		mesh->addRefPoint(3, 0, point5);
		mesh->addRefPoint(3, 1, point7);
		mesh->addRefPoint(3, 2, point7);
		mesh->addRefPoint(3, 3, point7);
		mesh->addRefPoint(3, 4, point5);

		mesh->addRefPoint(4, 0, point1);
		mesh->addRefPoint(4, 1, point2);
		mesh->addRefPoint(4, 2, point6);
		mesh->addRefPoint(4, 3, point2);
		mesh->addRefPoint(4, 4, point1);


		return mesh;
	}

	StateSpace initialConditions()
	{
		StateSpace initialConditions(SIRU3Model::NUM_DIMENSIONS);

		initialConditions[SIRU3Model::h_e] = -68.1355;
		initialConditions[SIRU3Model::h_i] = -77.2602;

		initialConditions[SIRU3Model::C_e] = 1;
		initialConditions[SIRU3Model::C_i] = 1;

		return initialConditions;
	}

	map<string, int> stateMap()
	{
		return SIRU3Model::stateMap();
	}

	map<string, int> paramMap()
	{
		return SIRU3Model::paramMap();
	}

	virtual string describe()
	{
		return "Hardcoded params";
	}
};


#endif /* SIRU3NONHOMOGENEOUSPARAMS_CUH_ */
