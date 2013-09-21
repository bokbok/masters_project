/*
 * SimulationRunner.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef SIMULATIONRUNNER_CUH_
#define SIMULATIONRUNNER_CUH_

#include "SimulationBuilder.cuh"
#include "StreamBuilder.cuh"
#include "params/Params.cuh"
#include "params/ParameterWriter.cuh"


const int REPORT_STEPS = 2000;
const int RENDER_STEPS = 2000;
const int MESH_SIZE = 50;
const double T_SIM = 60;
const double DELTA_T = 0.0000005;
const double DELTA = 0.1; //make smaller for tighter mesh
const double RANDOMISE_FRACTION = 0.01;

template <class T>
class SimulationRunner
{
private:

	Params<T> & _params;
	string _outputPath;

	std::map<string, int> dimensions()
	{
		std::map<string, int> dims;
		dims["h_e"] = T::h_e;
		dims["h_i"] = T::h_i;
		//		dims["T_ii"] = T::T_ii;
		//		dims["T_ie"] = T::T_ie;
		//		dims["T_ei"] = T::T_ei;
		//		dims["T_ee"] = T::T_ee;
		dims["C_e"] = T::C_e;
		dims["C_i"] = T::C_i;
		dims["phi_ee"] = T::phi_ee;
		dims["phi_ei"] = T::phi_ei;

		return dims;
	}

public:
	SimulationRunner(Params<T> & params, string outputPath) :
		_params(params),
		_outputPath(outputPath)
	{
	}

	void runSimulation()
	{
		StreamBuilder streamBuilder(MESH_SIZE, _outputPath);

		streamBuilder.toBinaryFile(dimensions())
				     .monitorConvergence();


		SimulationBuilder<T> simBuilder;
		ParameterMesh<T> * mesh = _params.mesh(MESH_SIZE);
		simBuilder.runFor(T_SIM)
				  .withTimeStep(DELTA_T)
				  .withMeshSize(MESH_SIZE)
				  .withMeshSpacing(DELTA)
				  .reportEvery(REPORT_STEPS)
				  .randomising(T::h_e)
				  .randomising(T::h_i)
				  .withInitialConditions(_params.initialConditions())
				  .withParameters(mesh)
				  .withICDeviation(RANDOMISE_FRACTION);


		ParameterWriter<T> paramWriter(T_SIM,
									   MESH_SIZE,
									   DELTA_T,
									   DELTA,
									   RANDOMISE_FRACTION,
									   &_params,
									   mesh,
									   streamBuilder.runPath());
		paramWriter.write();

		Simulation<T> * sim = simBuilder.build();

		sim->run(*streamBuilder.build());
	}
};


#endif /* SIMULATIONRUNNER_CUH_ */
