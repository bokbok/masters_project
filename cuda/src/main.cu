#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

#include "StateSpace.cuh"

#include "Mesh.cuh"
#include "liley/Model.cuh"
#include "liley/SIRU1Model.cuh"
#include "liley/SIRU2Model.cuh"
#include "io/FileDataStream.cuh"
#include "io/AsyncDataStream.cuh"
#include "io/CompositeDataStream.cuh"
#include "io/monitor/ConvergenceMonitor.cuh"
#include "io/visual/FrameRenderingDataStream.cuh"
#include "io/visual/TraceRenderingDataStream.cuh"
#include "io/visual/UITraceDataStream.cuh"

#include "Simulation.cuh"
#include <ctime>


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


ParameterSpace initialiseParamsSIRU2()
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
	params[SIRU2Model::e_i] = 1e-6;
	params[SIRU2Model::e_e] = 1e-6;

	return params;
}

ParameterSpace initialiseParamsSIRU1()
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

	params[SIRU1Model::mus_ee] = 0;
	params[SIRU1Model::mus_ei] = 0;
	params[SIRU1Model::mus_ie] = 0;
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



StateSpace initialConditionsBase()
{
	StateSpace initialConditions;

	initialConditions[Model::h_e] = -68.1355;
	initialConditions[Model::h_i] = -77.2602;

	return initialConditions;
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

	initialConditions[SIRU1Model::h_e] = 0;
	initialConditions[SIRU1Model::h_i] = 0;

	initialConditions[SIRU1Model::T_ee] = 0.3127;
	initialConditions[SIRU1Model::T_ei] = 0.9426;
	initialConditions[SIRU1Model::T_ie] = 0.4947;
	initialConditions[SIRU1Model::T_ii] = 1.4122;

	return initialConditions;
}



std::vector<int> dimensionsBase()
{
	std::vector<int> dims;
	dims.push_back(Model::h_e);
	dims.push_back(Model::h_i);

	return dims;
}

std::map<string, int> dimensionsSIRU1()
{
	std::map<string, int> dims;
	dims["h_e"] = SIRU1Model::h_e;
	dims["h_i"] = SIRU1Model::h_i;
	dims["T_ii"] = SIRU1Model::T_ii;
	dims["T_ie"] = SIRU1Model::T_ie;
	dims["T_ei"] = SIRU1Model::T_ei;
	dims["T_ee"] = SIRU1Model::T_ee;
	dims["phi_ee"] = SIRU1Model::phi_ee;
	dims["phi_ei"] = SIRU1Model::phi_ei;

	return dims;
}



typedef FileDataStream FileStream;

const int BUFFER_SIZE = 1000 / 25;
const int REPORT_STEPS = 200;
const int RENDER_STEPS = 400;
const int MESH_SIZE = 100;
const double T_SIM = 180;
const double DELTA_T = 0.000005;
const double DELTA = 0.2; //make smaller for tighter mesh
const double RANDOMISE_FRACTION = 0.01;
//const double RANDOMISE_FRACTION = 0;

const char * OUTPUT_PATH = "/terra/runs";

string runPath()
{
	double sysTime = time(0);

	char buf[600];
	sprintf(buf, "%i", (int) sysTime);

	mkdir(OUTPUT_PATH, 0777);
	return string(OUTPUT_PATH) + "/" + buf;
}


int main(void)
{
	srand(time(NULL));
	string path = runPath();
	mkdir(path.c_str(), 0777);

	FileStream file(path + "/run.dat", dimensionsSIRU1());
	AsyncDataStream fileOut(file);

	FrameRenderingDataStream renderer(path, MESH_SIZE, MESH_SIZE, SIRU1Model::h_e, RENDER_STEPS, initialiseParamsSIRU1()[SIRU1Model::h_e_rest]);
	AsyncDataStream renderOut(renderer);


	TraceRenderingDataStream heTrace(path, MESH_SIZE, MESH_SIZE, SIRU1Model::h_e, MESH_SIZE / 5, RENDER_STEPS, DELTA_T, -30, -80);
	AsyncDataStream heTraceOut(heTrace);

	TraceRenderingDataStream pspTrace(path, MESH_SIZE, MESH_SIZE, SIRU1Model::T_ii, MESH_SIZE / 2, RENDER_STEPS, DELTA_T, 0, 2);
	AsyncDataStream pspTraceOut(pspTrace);

//	UITraceDataStream uiTrace(MESH_SIZE, MESH_SIZE, SIRU1Model::h_e, MESH_SIZE / 2, RENDER_STEPS, DELTA_T, -30, -80);
//	AsyncDataStream uiTraceOut(uiTrace);

	ConvergenceMonitor monitor;
	AsyncDataStream convergenceStream(monitor);


	vector<DataStream *> streams;
	streams.push_back(&fileOut);
	streams.push_back(&renderOut);
	streams.push_back(&heTraceOut);
	streams.push_back(&pspTraceOut);
//	streams.push_back(&uiTraceOut);
	streams.push_back(&convergenceStream);

	CompositeDataStream out(streams);


	Simulation<SIRU1Model> sim(MESH_SIZE,
							  MESH_SIZE,
							  BUFFER_SIZE,
							  REPORT_STEPS,
							  T_SIM,
							  DELTA_T,
							  DELTA,
							  initialConditionsSIRU1(),
							  initialiseParamsSIRU1(),
							  RANDOMISE_FRACTION);

	sim.run(out);

	fileOut.waitToDrain();
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
