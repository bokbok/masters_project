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


const int REPORT_STEPS = 200;
const int RENDER_STEPS = 200;
const int MESH_SIZE = 70;
const double T_SIM = 60;
const double DELTA_T = 0.000002;
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

		double rmsMid = _params.params(MESH_SIZE)->paramsAt(0, 0)[T::h_e_rest];

		streamBuilder.toFile(dimensions())
				     .RMSFor(T::h_e, RENDER_STEPS, rmsMid)
				     .traceFor(T::h_e, RENDER_STEPS, 5, -80, -30)
				     //				     .traceFor(T::T_ii, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ie, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ei, RENDER_STEPS, 2, 0, 5)
				     //				     .traceFor(T::T_ee, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::C_e, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::C_i, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::phi_ee, RENDER_STEPS, 2, 0, 5)
				     .traceFor(T::phi_ei, RENDER_STEPS, 2, 0, 5)
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
				  .withParameters(_params.params(MESH_SIZE))
				  .withICDeviation(RANDOMISE_FRACTION);


		ParameterWriter<T> paramWriter(T_SIM,
									   MESH_SIZE,
									   DELTA_T,
									   DELTA,
									   RANDOMISE_FRACTION,
									   _params,
									   streamBuilder.runPath());
		paramWriter.write();

		Simulation<T> * sim = simBuilder.build();

		sim->run(*streamBuilder.build());
	}
};


#endif /* SIMULATIONRUNNER_CUH_ */
