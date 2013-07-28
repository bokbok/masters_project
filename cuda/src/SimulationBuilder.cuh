/*
 * SimulationFactory.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef SIMULATIONBUILDER_CUH_
#define SIMULATIONBUILDER_CUH_

#include "Simulation.cuh"
#include <vector>

using namespace std;

template <class T>
class SimulationBuilder
{
private:
	double _runtime;

	double _meshSpacing;
	double _timeStep;
	double _icFluctuation;

	int _meshSize;
	int _reportSteps;

	ParameterMesh<T> * _params;
	StateSpace _ics;

	vector<int> _randomiseParams;

public:
	SimulationBuilder()
	{

	}

	SimulationBuilder & runFor(double runtime)
	{
		_runtime = runtime;
		return *this;
	}

	SimulationBuilder & withMeshSize(int meshSize)
	{
		_meshSize = meshSize;
		return *this;
	}

	SimulationBuilder & withMeshSpacing(double meshSpacing)
	{
		_meshSpacing = meshSpacing;
		return *this;
	}

	SimulationBuilder & withTimeStep(double timeStep)
	{
		_timeStep = timeStep;
		return *this;
	}

	SimulationBuilder & reportEvery(int reportSteps)
	{
		_reportSteps = reportSteps;
		return *this;
	}

	SimulationBuilder & withParameters(ParameterMesh<T> * params)
	{
		_params = params;
		return *this;
	}

	SimulationBuilder & randomising(int randomiseParam)
	{
		_randomiseParams.push_back(randomiseParam);
		return *this;
	}

	SimulationBuilder & withICDeviation(double icFluctuation)
	{
		_icFluctuation = icFluctuation;
		return *this;
	}

	SimulationBuilder & withInitialConditions(StateSpace ics)
	{
		_ics = ics;
		return *this;
	}

	Simulation<T> * build()
	{
		Simulation<T> * sim = new Simulation<T>(_meshSize, _meshSize, 200, _reportSteps, _runtime,
								  	      	 	 _timeStep, _meshSpacing, _ics,
								  	      	 	 *_params, _icFluctuation, _randomiseParams);

		return sim;
	}
};


#endif /* SIMULATIONBUILDER_CUH_ */
