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


const int REPORT_STEPS = 200;
const int RENDER_STEPS = 200;
const int MESH_SIZE = 10;
const double T_SIM = 20;
const double DELTA_T = 0.000002;
const double DELTA = 0.1; //make smaller for tighter mesh
const double RANDOMISE_FRACTION = 0.001;

template <class T>
class SimulationRunner
{
private:

	Params & _params;
	string _outputPath;

	std::map<string, int> dimensions()
	{
		std::map<string, int> dims;
		dims["h_e"] = T::h_e;
		dims["h_i"] = T::h_i;
		dims["T_ii"] = T::T_ii;
		dims["T_ie"] = T::T_ie;
		dims["T_ei"] = T::T_ei;
		dims["T_ee"] = T::T_ee;
		dims["phi_ee"] = T::phi_ee;
		dims["phi_ei"] = T::phi_ei;

		return dims;
	}

public:
	SimulationRunner(Params & params, string outputPath) :
		_params(params),
		_outputPath(outputPath)
	{
	}

	void runSimulation()
	{
		StreamBuilder streamBuilder(MESH_SIZE, _outputPath);

		streamBuilder.toFile(dimensions())
				     .RMSFor(T::h_e, RENDER_STEPS, _params.params()[T::h_e_rest])
				     .traceFor(T::h_e, RENDER_STEPS, 5, -80, -30)
				     //				     .traceFor(T::T_ii, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ie, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ei, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ee, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::C_e, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::C_i, RENDER_STEPS, 2, 0, 5)
				     .monitorConvergence();


		SimulationBuilder<T> simBuilder;

		simBuilder.runFor(T_SIM)
				  .withTimeStep(DELTA_T)
				  .withMeshSize(MESH_SIZE)
				  .withMeshSpacing(DELTA)
				  .reportEvery(REPORT_STEPS)
				  .randomising(T::h_e)
				  .randomising(T::h_i)
				  .withInitialConditions(_params.initialConditions())
				  .withParameters(_params.params())
				  .withICDeviation(RANDOMISE_FRACTION);


		Simulation<T> * sim = simBuilder.build();

		sim->run(*streamBuilder.build());
	}
};


#endif /* SIMULATIONRUNNER_CUH_ */
