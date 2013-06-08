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

class Derivatives
{
private:
	double _vals[MAX_EQUATIONS];
	int _numEquations;

	__device__ __host__
	void construct()
	{
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			_vals[i] = 0.0;
		}
	}

public:
	__device__ __host__
	Derivatives(int numEquations = MAX_EQUATIONS) :
		_numEquations(numEquations)
	{
		construct();
	}

	__device__
	double & operator [](int index)
	{
		CHECK_BOUNDS(index, _numEquations);
		return _vals[index];
	}

	__device__
	StateSpace operator *(double mult)
	{
		StateSpace res(_numEquations);

		for (int i = 0; i < _numEquations; i++)
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
