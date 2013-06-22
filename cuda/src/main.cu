#include "common.cuh"

using namespace std;

#include "liley/SIRU1Model.cuh"
#include "liley/SIRU2Model.cuh"

#include "params/SIRU1HardcodedParams.cuh"
#include "params/SIRU2HardcodedParams.cuh"
#include "SimulationRunner.cuh"

const char * OUTPUT_PATH = "/terra/runs";

int main(void)
{
	SIRU2HardcodedParams params;
	SimulationRunner<SIRU2Model> runner(params, OUTPUT_PATH);
	runner.runSimulation();

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
