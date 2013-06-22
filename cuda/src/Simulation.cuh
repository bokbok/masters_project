/*
 * Simulation.hpp
 *
 *  Created on: 21/05/2013
 *      Author: matt
 */

#ifndef SIMULATION_HPP_
#define SIMULATION_HPP_
#include "Mesh.cuh"

template <class T>
class Simulation
{
private:
	int _width, _height;
	double _length, _deltaT, _delta, _icFluctuation;
	StateSpace _initialConditions;
	ParameterSpace _params;

	Mesh<T> * _mesh;

	StateSpace * randomiseInitialConditions(vector<int> & randomiseParams)
	{
		StateSpace * result = new StateSpace[_width * _height];

		for (int x = 0; x < _width; x++)
		{
			for (int y = 0; y < _height; y++)
			{
				result[x + y * _width].randomise(_icFluctuation, _initialConditions, randomiseParams);
			}
		}

		return result;
	}

public:
	Simulation(int width, int height, int bufferSize, int reportSteps, double length,
			   double deltaT, double delta, StateSpace initialConditions,
			   ParameterSpace params, double icFluctuation, vector<int> & randomiseParams):
		_width(width),
		_height(height),
		_length(length),
		_deltaT(deltaT),
		_delta(delta),
		_initialConditions(initialConditions),
		_params(params),
		_icFluctuation(icFluctuation)
	{
		printf("Allocating mesh......");
		_mesh = new Mesh<T>(width, height, delta, bufferSize, reportSteps, randomiseInitialConditions(randomiseParams), params);
		printf("Done......");
	}

	~Simulation()
	{
		delete _mesh;
	}

	void run(DataStream & out)
	{
		printf("Starting simulation......\n");
		double deltaTMs = _deltaT / 1e-3;
		for (int i = 0; i < _length / _deltaT; i++)
		{
			_mesh->stepAndFlush(i * _deltaT, deltaTMs, out);
		}
	}

};


#endif /* SIMULATION_HPP_ */
