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
#include <dirent.h>

const char * OUTPUT_PATH = "/terra/runs";
//const char * PARAM_DIR = "/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp41.ode";
const char * PARAM_DIR = "/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp79.ode";

vector<string> parameterFiles(const char * dir)
{
	vector<string> result;
	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(dir);
	if (dpdf != NULL){
	   while (epdf = readdir(dpdf)){
		  string file = epdf->d_name;
		  if (file.length() > 4)
		  {
			  if (file.compare(file.length() - 4, file.length(), ".yml") == 0)
			  {
				  cout << string(dir) + string(epdf->d_name) << endl;
				  result.push_back(string(dir) + "/" + string(epdf->d_name));
			  }
		  }
	   }
	}

	return result;
}

int main(void)
{
	setbuf(stdout, NULL);
	//SIRU3HardcodedParams params;
	//SIRU3NonHomogeneousParams params;

//	YAMLModelParams<SIRU3Model> params("/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp41.ode/1375611399.98.yml");

	NonHomogeneousYAMLModelParams<SIRU3Model> params(parameterFiles(PARAM_DIR), 17);

	SimulationRunner<SIRU3Model> runner(params, OUTPUT_PATH);
	runner.runSimulation();

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
