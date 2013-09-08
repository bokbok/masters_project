/*
 * PartitionedParameterMesh.cuh
 *
 *  Created on: 29/08/2013
 *      Author: matt
 */

#ifndef PARTITIONEDPARAMETERMESH_CUH_
#define PARTITIONEDPARAMETERMESH_CUH_

#include "ParameterMesh.cuh"
#include <vector>

template <class M>
class PartitionedParameterMesh : public ParameterMesh<M>
{
private:
	vector<ParameterSpace> _params;
	int _meshSize;
	float _regionSize, _N;

	int region(int x, int y)
	{
		return floor(x * _regionSize) + floor(y * _regionSize) * _N;
	}

public:
	PartitionedParameterMesh(vector<ParameterSpace> params, int meshSize):
		_params(params),
		_meshSize(meshSize)
	{
		_N = sqrt(_params.size());
		_regionSize = _N / meshSize;
	}

	virtual ~PartitionedParameterMesh()
	{
	}

	virtual ParameterSpace paramsAt(int x, int y)
	{
		return _params[region(y, x)];
	}


};


#endif /* PARTITIONEDPARAMETERMESH_CUH_ */
