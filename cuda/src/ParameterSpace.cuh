/*
 * ParameterSpace.cuh
 *
 *  Created on: 01/05/2013
 *      Author: matt
 */

#ifndef PARAMETERSPACE_CUH_
#define PARAMETERSPACE_CUH_


const int MAX_PARAMS = 100;

class __align__(128)
		ParameterSpace
{
private:
	double _vals[MAX_PARAMS];

public:
	__device__ __host__
	ParameterSpace()
	{
		for (int i = 0; i < MAX_PARAMS; i++)
		{
			_vals[i] = 0;
		}
	}

	__device__ __host__
	inline double & operator [](int index)
	{
		CHECK_BOUNDS(index, MAX_PARAMS);
		return _vals[index];
	}

	__device__
	operator double *()
	{
		return _vals;
	}


};


#endif /* PARAMETERSPACE_CUH_ */
