/*
 * YAMLModelParams.cuh
 *
 *  Created on: 04/08/2013
 *      Author: matt
 */

#ifndef YAMLMODELPARAMS_CUH_
#define YAMLMODELPARAMS_CUH_

#include "Params.cuh"
#include "HomogeneousParameterMesh.cuh"
#include "yaml/YAMLParams.hpp"
#include <iostream>

template <class M>
class YAMLModelParams : public Params<M>
{
private:
	YAMLParams _params;
	ParameterSpace _prototype;

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

public:
	YAMLModelParams(string filename) :
		_params(filename)
	{
		readPrototype();
	}

	ParameterMesh<M> * mesh(int meshSize)
	{
		HomogeneousParameterMesh<M> * mesh = new HomogeneousParameterMesh<M>(_prototype);
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

	virtual string describe()
	{
		return "Params from file: " + _params.filename();
	}


};


#endif /* YAMLMODELPARAMS_CUH_ */
