/*
 * StateSpace.cuh
 *
 *  Created on: 29/04/2013
 *      Author: matt
 */

#ifndef STATESPACE_CUH_
#define STATESPACE_CUH_

__device__ __host__
struct StateSpace
{
	double * _stateVars;
	int _dims;

	StateSpace()
	{

	}

	double * stateVars()
	{
		return _stateVars;
	}


	int dims()
	{
		return _dims;
	}

	static StateSpace zero;
};


#endif /* STATESPACE_CUH_ */
