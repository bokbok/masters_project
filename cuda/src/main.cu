#include "common.cuh"

using namespace std;

#include "liley/SIRU1Model.cuh"
#include "liley/SIRU2Model.cuh"
#include "liley/SIRU3Model.cuh"

#include "params/SIRU1HardcodedParams.cuh"
#include "params/SIRU2HardcodedParams.cuh"
#include "params/SIRU3HardcodedParams.cuh"
#include "params/SIRU3NonHomogeneousParams.cuh"
#include "SimulationRunner.cuh"

const char * OUTPUT_PATH = "/terra/runs";

int main(void)
{
	setbuf(stdout, NULL);
	//SIRU3HardcodedParams params;
	SIRU3NonHomogeneousParams params;
	SimulationRunner<SIRU3Model> runner(params, OUTPUT_PATH);
	runner.runSimulation();

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
