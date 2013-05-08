#include <stdio.h>

using namespace std;

#include "StateSpace.cuh"

#include "Mesh.cuh"
#include "liley/Model.cuh"

ParameterSpace initialiseParams()
{
	ParameterSpace params;
	params[Model::tor_e] = 138.3660;
	params[Model::tor_i] = 89.3207;
	params[Model::h_e_rest] = -68.1355;
	params[Model::h_i_rest] = -77.2602;

	params[Model::h_ee_eq] = -15.8527;
	params[Model::h_ei_eq] = 7.4228;
	params[Model::h_ie_eq] = -85.9896;
	params[Model::h_ii_eq] = -84.5363;

	params[Model::gamma_ee] = 0.4393;
	params[Model::gamma_ei] = 0.2350;
	params[Model::gamma_ie] = 0.0791;
	params[Model::gamma_ii] = 0.0782;

	params[Model::T_ee] = 0.3127;
	params[Model::T_ei] = 0.9426;
	params[Model::T_ie] = 0.4947;
	params[Model::T_ii] = 1.4122;

	params[Model::N_beta_ee] = 4582.0661;
	params[Model::N_beta_ei] = 4198.1829;
	params[Model::N_beta_ie] = 989.5281;
	params[Model::N_beta_ii] = 531.9419;

	params[Model::N_alpha_ee] = 4994.4860;
	params[Model::N_alpha_ei] = 2222.9060;

	params[Model::s_e_max] = 0.2801;
	params[Model::s_i_max] = 0.1228;

	params[Model::mu_e] = -47.1364;
	params[Model::mu_i] = -45.3751;

	params[Model::sigma_e] = 2.6120;
	params[Model::sigma_i] = 2.8294;

	params[Model::p_ee] = 3.6032;
	params[Model::p_ei] = 0.3639;
	params[Model::p_ie] = 0;
	params[Model::p_ii] = 0;

	params[Model::phi_ie] = 0;
	params[Model::phi_ii] = 0;

	params[Model::A_ee] = 0.2433;
	params[Model::A_ei] = 0.2433;

	params[Model::v] = 0.1714;

	params[Model::r_abs] = 0;

	return params;
}

StateSpace initialConditions()
{
	StateSpace initialConditions;

	initialConditions[Model::h_e] = -22.1355;
	initialConditions[Model::h_i] = -22.2602;

	return initialConditions;
}

const int STEPS = 1000;
int main(void)
{
	size_t limit;
	cudaDeviceGetLimit(&limit, cudaLimitStackSize);
	printf(">>>>>> Stacksize % i \n\n\n\n", limit);
	cudaDeviceSetLimit(cudaLimitStackSize, limit * 2);
	Mesh mesh(100, 100, 0.0001, STEPS, initialConditions(), initialiseParams());

	double deltaT = 0.0001;

	for (int i = 0; i < 100 * STEPS; i++)
	{
		mesh.stepAndFlush(i * deltaT, deltaT, cout);
	}

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
