/*
 * Derivatives.cuh
 *
 *  Created on: 03/05/2013
 *      Author: matt
 */

#ifndef DERIVATIVES_CUH_
#define DERIVATIVES_CUH_

#include "common.cuh"
#include "StateSpace.cuh"
//const int MAX_EQUATIONS=40;

class Derivatives
{
private:
	double _vals[MAX_EQUATIONS];
	int _equationCount;

public:
	__device__ __host__
	Derivatives()
	{
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			_vals[i] = 0.0;
		}
	}

	__device__
	double & operator [](int index)
	{
		CHECK_BOUNDS(index, MAX_EQUATIONS);
		return _vals[index];
	}

	__device__
	StateSpace operator *(double mult)
	{
		StateSpace res;

		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			res[i] = _vals[i] * mult;
		}

		return res;
	}

	__device__
	bool debugNanCheck()
	{
		bool statenan = false;
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			statenan |= isnan(_vals[i]);
		}

		return statenan;
	}
};


#endif /* DERIVATIVES_CUH_ */
