/*
 * NonHomogeneousYAMLModelParams.cuh
 *
 *  Created on: 04/08/2013
 *      Author: matt
 */

#ifndef NONHOMOGENEOUSYAMLMODELPARAMS_CUH_
#define NONHOMOGENEOUSYAMLMODELPARAMS_CUH_

#include "Params.cuh"
#include "NonHomogeneousParameterMesh.cuh"
#include "yaml/YAMLParams.hpp"
#include <iostream>
#include <vector>

template <class M>
class NonHomogeneousYAMLModelParams : public Params<M>
{
	const static int SAMPLE_POINTS = 5;

private:
	YAMLParams _params;
	ParameterSpace _prototype;
	vector<string> _randomise;

	void readPrototype()
	{
		map<string, double> readParams = _params.read();
		map<string, int> paramLookup = paramMap();

		for (map<string, double>::iterator iter = readParams.begin(); iter != readParams.end(); ++iter)
		{
			if (paramLookup.find(iter->first) != paramLookup.end())
			{
				cout << iter->first << "=" << iter->second << endl;
				_prototype[paramLookup[iter->first]] = iter->second;
			}
		}
	}

	double randAmount()
	{
		return 1 + (rand() % 1000) * 0.1 / 1000;
	}

public:
	NonHomogeneousYAMLModelParams(string filename, vector<string> randomise) :
		_params(filename),
		_randomise(randomise)
	{
		readPrototype();
	}

	ParameterMesh<M> * params(int meshSize)
	{
		NonHomogeneousParameterMesh<M> * mesh = new NonHomogeneousParameterMesh<M>(SAMPLE_POINTS, meshSize);

		ParameterSpace point1 = _prototype;
		ParameterSpace point2 = _prototype;
		ParameterSpace point3 = _prototype;
		ParameterSpace point4 = _prototype;
		ParameterSpace point5 = _prototype;
		ParameterSpace point6 = _prototype;
		ParameterSpace point7 = _prototype;
		ParameterSpace point8 = _prototype;

		for (vector<string>::iterator iter = _randomise.begin(); iter != _randomise.end(); ++iter)
		{
			int dim = paramMap()[*iter];
			point1[dim] *= randAmount();
			point2[dim] *= randAmount();
			point3[dim] *= randAmount();
			point4[dim] *= randAmount();
			point5[dim] *= randAmount();
			point6[dim] *= randAmount();
			point7[dim] *= randAmount();
			point8[dim] *= randAmount();
		}


		mesh->addRefPoint(0, 0, point1);
		mesh->addRefPoint(0, 1, point2);
		mesh->addRefPoint(0, 2, point6);
		mesh->addRefPoint(0, 3, point2);
		mesh->addRefPoint(0, 4, point1);

		mesh->addRefPoint(1, 0, point3);
		mesh->addRefPoint(1, 1, point4);
		mesh->addRefPoint(1, 2, point4);
		mesh->addRefPoint(1, 3, point4);
		mesh->addRefPoint(1, 4, point3);

		mesh->addRefPoint(2, 0, point5);
		mesh->addRefPoint(2, 1, point7);
		mesh->addRefPoint(2, 2, point8); // centre
		mesh->addRefPoint(2, 3, point7);
		mesh->addRefPoint(2, 4, point5);

		mesh->addRefPoint(3, 0, point5);
		mesh->addRefPoint(3, 1, point7);
		mesh->addRefPoint(3, 2, point7);
		mesh->addRefPoint(3, 3, point7);
		mesh->addRefPoint(3, 4, point5);

		mesh->addRefPoint(4, 0, point1);
		mesh->addRefPoint(4, 1, point2);
		mesh->addRefPoint(4, 2, point6);
		mesh->addRefPoint(4, 3, point2);
		mesh->addRefPoint(4, 4, point1);

		return mesh;
	}

	StateSpace initialConditions()
	{
		StateSpace initialConditions(M::NUM_DIMENSIONS);

		initialConditions[M::h_e] = _prototype[M::h_e_rest];
		initialConditions[M::h_i] = _prototype[M::h_i_rest];

		initialConditions[M::C_e] = 1;
		initialConditions[M::C_i] = 1;

		return initialConditions;
	}

	map<string, int> stateMap()
	{
		return M::stateMap();
	}

	map<string, int> paramMap()
	{
		return M::paramMap();
	}

};


#endif /* NONHOMOGENEOUSYAMLMODELPARAMS_CUH_ */
