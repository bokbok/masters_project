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
#include "params/PartitionedYAMLParams.cuh"
#include "params/ShapePartitionedParams.cuh"
#include "params/shapes/CircularPartition.cuh"

#include "SimulationRunner.cuh"
#include <dirent.h>

const char * OUTPUT_PATH = "/Users/matt/work/study/runs";
//const char * PARAM_DIR = "/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp41.ode";
//const char * PARAM_DIR = "/home/matt/work/masters_project/parameterisations/derived/parameterisations/original_biphasic_86.yml/bp79.ode";

const char * PARAM_DIR = "/Users/matt/work/study/masters_project/parameterisations/partitioned/set11";

//const char * PARAM_DIR = "/home/matt/work/masters_project/parameterisations/partitioned/set11";

vector<string> parameterFiles(const char * dir)
{
	vector<string> result;
	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(dir);
	if (dpdf != NULL){
	   while (epdf = readdir(dpdf))
	   {
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

void reportMemoryStats()
{
	// show memory usage of GPU

	size_t free_byte ;
	size_t total_byte ;

	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

	if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
	}



	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

int main(void)
{
	reportMemoryStats();
	setbuf(stdout, NULL);
	//PartitionedYAMLParams<SIRU3Model> params(parameterFiles(PARAM_DIR));

	string paramDir = PARAM_DIR;

	ShapePartitionedParams<SIRU3Model> params(paramDir + "/partition1.yml");
	params.addPartition(new CircularPartition<SIRU3Model>(paramDir + "/partition2.yml", 0.25, 0.25, 0.05, 0.2));
	params.addPartition(new CircularPartition<SIRU3Model>(paramDir + "/partition3.yml", 0.5, 0.5, 0.05, 0.2));
	params.addPartition(new CircularPartition<SIRU3Model>(paramDir + "/partition4.yml", 0.75, 0.75, 0.05, 0.2));
	params.addPartition(new CircularPartition<SIRU3Model>(paramDir + "/partition5.yml", 0.25, 0.75, 0.05, 0.2));
	params.addPartition(new CircularPartition<SIRU3Model>(paramDir + "/partition6.yml", 0.75, 0.25, 0.05, 0.2));


	SimulationRunner<SIRU3Model> runner(params, OUTPUT_PATH);
	runner.runSimulation();

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
