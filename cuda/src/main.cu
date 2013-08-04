#include "common.cuh"

using namespace std;

#include "liley/SIRU1Model.cuh"
#include "liley/SIRU2Model.cuh"
#include "liley/SIRU3Model.cuh"

#include "params/SIRU1HardcodedParams.cuh"
#include "params/SIRU2HardcodedParams.cuh"
#include "params/SIRU3HardcodedParams.cuh"
#include "params/SIRU3NonHomogeneousParams.cuh"
#include "params/YAMLModelParams.cuh"
#include "params/NonHomogeneousYAMLModelParams.cuh"
#include "SimulationRunner.cuh"

const char * OUTPUT_PATH = "/terra/runs";
const char * PARAM_FILE = "/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp41.ode/1375611399.98.yml";

int main(void)
{
	setbuf(stdout, NULL);
	//SIRU3HardcodedParams params;
	//SIRU3NonHomogeneousParams params;

//	YAMLModelParams<SIRU3Model> params("/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp41.ode/1375611399.98.yml");

	vector<string> randomise;
	randomise.push_back("mus_i");
	randomise.push_back("mus_e");
	randomise.push_back("e_ii");
	randomise.push_back("e_ie");

	NonHomogeneousYAMLModelParams<SIRU3Model> params(PARAM_FILE, randomise);

	SimulationRunner<SIRU3Model> runner(params, OUTPUT_PATH);
	runner.runSimulation();

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
