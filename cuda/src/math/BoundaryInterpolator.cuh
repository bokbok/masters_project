/*
 * BoundaryInterpolator.cuh
 *
 *  Created on: 09/09/2013
 *      Author: matt
 */

#ifndef BOUNDARYINTERPOLATOR_CUH_
#define BOUNDARYINTERPOLATOR_CUH_

#include "../ParameterSpace.cuh"

class BoundaryInterpolator
{
private:
	double _boundarySize;
	ParameterSpace _baseVals;
	ParameterSpace _regionVals;
	ParameterSpace _delta;

public:
	BoundaryInterpolator(double boundarySize, ParameterSpace baseVals, ParameterSpace regionVals):
		_boundarySize(boundarySize),
		_baseVals(baseVals),
		_regionVals(regionVals),
		_delta(_regionVals - _baseVals)
	{
	}

	ParameterSpace interpolate(double dist)
	{
		ParameterSpace result = _baseVals + _delta * (dist / _boundarySize);

		return result;
	}

};


#endif /* BOUNDARYINTERPOLATOR_CUH_ */
