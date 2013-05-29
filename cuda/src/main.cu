#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

#include "StateSpace.cuh"

#include "Mesh.cuh"
#include "liley/Model.cuh"
#include "liley/SIRUModel.cuh"
#include "io/FileDataStream.cuh"
#include "io/MemoryMappedFileDataStream.cuh"
#include "io/AsyncDataStream.cuh"
#include "io/CompositeDataStream.cuh"
#include "io/visual/FrameRenderingDataStream.cuh"

#include "Simulation.cuh"

ParameterSpace initialiseParamsBase()
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


ParameterSpace initialiseParamsSIRU()
{
	ParameterSpace params;
	params[SIRUModel::tor_e] = 138.3660;
	params[SIRUModel::tor_i] = 89.3207;
	params[SIRUModel::h_e_rest] = -68.1355;
	params[SIRUModel::h_i_rest] = -77.2602;

	params[SIRUModel::h_ee_eq] = -15.8527;
	params[SIRUModel::h_ei_eq] = 7.4228;
	params[SIRUModel::h_ie_eq] = -85.9896;
	params[SIRUModel::h_ii_eq] = -84.5363;

	params[SIRUModel::gamma_ee] = 0.4393;
	params[SIRUModel::gamma_ei] = 0.2350;
	params[SIRUModel::gamma_ie] = 0.0791;
	params[SIRUModel::gamma_ii] = 0.0782;

	params[SIRUModel::T_ee_p] = 0.3127;
	params[SIRUModel::T_ei_p] = 0.9426;
	params[SIRUModel::T_ie_p] = 0.4947;
	params[SIRUModel::T_ii_p] = 1.4122;

	params[SIRUModel::N_beta_ee] = 4582.0661;
	params[SIRUModel::N_beta_ei] = 4198.1829;
	params[SIRUModel::N_beta_ie] = 989.5281;
	params[SIRUModel::N_beta_ii] = 531.9419;

	params[SIRUModel::N_alpha_ee] = 4994.4860;
	params[SIRUModel::N_alpha_ei] = 2222.9060;

	params[SIRUModel::s_e_max] = 0.2801;
	params[SIRUModel::s_i_max] = 0.1228;

	params[SIRUModel::mu_e] = -47.1364;
	params[SIRUModel::mu_i] = -45.3751;

	params[SIRUModel::mus_e] = 0.001;
	params[SIRUModel::mus_i] = 0.001;

	params[SIRUModel::sigma_e] = 2.6120;
	params[SIRUModel::sigma_i] = 2.8294;

	params[SIRUModel::p_ee] = 3.6032;
	params[SIRUModel::p_ei] = 0.3639;
	params[SIRUModel::p_ie] = 0;
	params[SIRUModel::p_ii] = 0;

	params[SIRUModel::phi_ie] = 0;
	params[SIRUModel::phi_ii] = 0;

	params[SIRUModel::A_ee] = 0.2433;
	params[SIRUModel::A_ei] = 0.2433;

	params[SIRUModel::v] = 0.1714;

	params[SIRUModel::r_abs] = 0;

	params[SIRUModel::k_i] = 0.1;
	params[SIRUModel::k_e] = 0.2;
	params[SIRUModel::e_i] = 1.8;
	params[SIRUModel::e_e] = 1.5;

	return params;
}


StateSpace initialConditionsBase()
{
	StateSpace initialConditions;

	initialConditions[Model::h_e] = -68.1355;
	initialConditions[Model::h_i] = -77.2602;

	return initialConditions;
}

StateSpace initialConditionsSIRU()
{
	StateSpace initialConditions;

	initialConditions[SIRUModel::h_e] = 0;
	initialConditions[SIRUModel::h_i] = 0;

	initialConditions[SIRUModel::T_ee] = 0.3127;
	initialConditions[SIRUModel::T_ei] = 0.9426;
	initialConditions[SIRUModel::T_ie] = 0.4947;
	initialConditions[SIRUModel::T_ii] = 1.4122;

	return initialConditions;
}



std::vector<int> dimensionsBase()
{
	std::vector<int> dims;
	dims.push_back(Model::h_e);
	dims.push_back(Model::h_i);

	return dims;
}

std::map<string, int> dimensionsSIRU()
{
	std::map<string, int> dims;
	dims["h_e"] = SIRUModel::h_e;
	dims["h_i"] = SIRUModel::h_i;
	dims["T_ii"] = SIRUModel::T_ii;
	dims["T_ie"] = SIRUModel::T_ie;
	dims["T_ei"] = SIRUModel::T_ei;
	dims["T_ee"] = SIRUModel::T_ee;
	dims["phi_ee"] = SIRUModel::phi_ee;
	dims["phi_ei"] = SIRUModel::phi_ei;

	return dims;
}

typedef FileDataStream FileStream;

const int BUFFER_SIZE = 1000 / 25;
const int REPORT_STEPS = 100;
const int MESH_SIZE = 100;
const double T_SIM = 120;
const double DELTA_T = 0.0001;
const double DELTA = 10;
const double RANDOMISE_FRACTION = 1e-4;

//const char * OUTPUT_PATH = "/var/tmp/run.dat";
const char * OUTPUT_PATH = "/terra/run.dat";
const char * RENDER_PATH = "/terra/run_images";

int main(void)
{
	FileStream file(OUTPUT_PATH, dimensionsSIRU());
	AsyncDataStream fileOut(file);

	FrameRenderingDataStream renderer(RENDER_PATH, MESH_SIZE, MESH_SIZE, SIRUModel::h_e, 10);
	AsyncDataStream renderOut(renderer);

	vector<DataStream *> streams;
	streams.push_back(&fileOut);
	streams.push_back(&renderOut);

	CompositeDataStream out(streams);


	Simulation<SIRUModel> sim(MESH_SIZE, MESH_SIZE, BUFFER_SIZE, REPORT_STEPS, T_SIM, DELTA_T, DELTA, initialConditionsSIRU(), initialiseParamsSIRU(), RANDOMISE_FRACTION);

	sim.run(out);

	fileOut.waitToDrain();
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
