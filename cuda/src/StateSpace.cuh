/*
 * StateSpace.cuh
 *
 *  Created on: 29/04/2013
 *      Author: matt
 */

#ifndef STATESPACE_CUH_
#define STATESPACE_CUH_

#include "common.cuh"

#include "ParameterSpace.cuh"
const int MAX_EQUATIONS=40;
class StateSpace
{
private:
	double _vals[MAX_EQUATIONS];

public:
	__device__ __host__
	StateSpace()
	{
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			_vals[i] = 0.0;
		}
	}

	__device__ __host__
	double & operator [](int index)
	{
		CHECK_BOUNDS(index, MAX_EQUATIONS);
		return _vals[index];
	}

	__device__
	StateSpace operator /(double val)
	{
		StateSpace res;

		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			res[i] = _vals[i] / val;
		}

		return res;
	}

	__device__
	StateSpace operator *(double val)
	{
		StateSpace res;

		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			res[i] = _vals[i] * val;
		}

		return res;
	}

	__device__
	void update(StateSpace & val)
	{
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			_vals[i] = val[i];
		}
	}

	__device__
	StateSpace operator +(StateSpace &rhs)
	{
		StateSpace result;
		for (int i = 0; i < MAX_EQUATIONS; i++)
		{
			result[i] = _vals[i] + rhs[i];
		}
		return result;

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

#endif /* STATESPACE_CUH_ */
