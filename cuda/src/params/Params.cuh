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

class Params
{
public:
	virtual ParameterSpace params() = 0;
	virtual StateSpace initialConditions() = 0;

	virtual ~Params() {};
};


#endif /* PARAMS_CUH_ */
