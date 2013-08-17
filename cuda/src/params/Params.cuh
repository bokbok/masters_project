/*
 * Params.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef PARAMS_CUH_
#define PARAMS_CUH_

#include "../ParameterSpace.cuh"
#include "../StateSpace.cuh"
#include "ParameterMesh.cuh"
#include <fstream>
#include <string>

using namespace std;

template <class T>
class Params
{
public:
	virtual ParameterMesh<T> * mesh(int meshSize) = 0;
	virtual StateSpace initialConditions() = 0;
	virtual map<string, int> paramMap() = 0;
	virtual map<string, int> stateMap() = 0;

	virtual ~Params() {}
};


#endif /* PARAMS_CUH_ */
