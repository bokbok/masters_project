/*
 * StateSpace.cuh
 *
 *  Created on: 29/04/2013
 *      Author: matt
 */

#ifndef STATESPACE_CUH_
#define STATESPACE_CUH_

#include "common.cuh"
#include <math.h>

#include "ParameterSpace.cuh"
const int MAX_EQUATIONS=40;

class __align__(128) StateSpace
{
private:
	double _vals[MAX_EQUATIONS];
	double _t;
	int _numDimensions;

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
	StateSpace(int numDimensions = MAX_EQUATIONS) : _numDimensions(numDimensions)
	{
		construct();
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

	__device__ __host__
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

	__device__ __host__
	operator double *()
	{
		return _vals;
	}

	__device__
	StateSpace operator /(double val)
	{
		StateSpace res(_numDimensions);

		for (int i = 0; i < _numDimensions; i++)
		{
			res[i] = _vals[i] / val;
		}

		return res;
	}

	__device__
	StateSpace operator *(double val)
	{
		StateSpace res(_numDimensions);

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
	void update(double t, double * val)
	{
		_t = t;
		for (int i = 0; i < _numDimensions; i++)
		{
			_vals[i] = val[i];
		}
	}

	__device__
	void setT(double t)
	{
		_t = t;
	}

	__host__
	void randomise(double deviation, StateSpace & from)
	{
		_numDimensions = from._numDimensions;
		for (int i = 0; i < _numDimensions; i++)
		{
			_vals[i] = from[i] * (1 + random() * deviation);
		}
	}

	__host__
	double random()
	{
		return (double)(rand() % 200 - 100) / 100;
	}

	__device__
	StateSpace operator +(StateSpace &rhs)
	{
		StateSpace result(_numDimensions);
		for (int i = 0; i < _numDimensions; i++)
		{
			result[i] = _vals[i] + rhs[i];
		}
		return result;

	}

	__device__ __host__
	bool nan()
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
