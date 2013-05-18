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
	double _t;
	int _numDimensions;

public:
	__device__ __host__
	StateSpace()
	{
		_numDimensions = MAX_EQUATIONS;
		for (int i = 0; i < _numDimensions; i++)
		{
			_vals[i] = 0.0;
		}
	}

	__host__
	double * data()
	{
		return _vals;
	}

	__host__
	double t()
	{
		return _t;
	}

	__host__
	int numDimensions()
	{
		return _numDimensions;
	}

	__device__ __host__
	double & operator [](int index)
	{
		CHECK_BOUNDS(index, _numDimensions);
		return _vals[index];
	}

	__device__
	StateSpace operator /(double val)
	{
		StateSpace res;

		for (int i = 0; i < _numDimensions; i++)
		{
			res[i] = _vals[i] / val;
		}

		return res;
	}

	__device__
	StateSpace operator *(double val)
	{
		StateSpace res;

		for (int i = 0; i < _numDimensions; i++)
		{
			res[i] = _vals[i] * val;
		}

		return res;
	}

	__device__
	void update(double t, StateSpace & val)
	{
		_t = t;
		for (int i = 0; i < _numDimensions; i++)
		{
			_vals[i] = val[i];
		}
	}

	__device__
	StateSpace operator +(StateSpace &rhs)
	{
		StateSpace result;
		for (int i = 0; i < _numDimensions; i++)
		{
			result[i] = _vals[i] + rhs[i];
		}
		return result;

	}

	__device__
	bool debugNanCheck()
	{
		bool statenan = false;
		for (int i = 0; i < _numDimensions; i++)
		{
			statenan |= isnan(_vals[i]);
		}

		return statenan;
	}
};

#endif /* STATESPACE_CUH_ */
