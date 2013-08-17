/*
 * NonHomogeneousParameterMesh.cuh
 *
 *  Created on: 25/07/2013
 *      Author: matt
 */

#ifndef NONHOMOGENEOUSPARAMETERMESH_CUH_
#define NONHOMOGENEOUSPARAMETERMESH_CUH_

#include <map>
#include <vector>
#include <algorithm>
#include "ParameterMesh.cuh"
#include "../math/Interpolator.hpp"

using namespace std;

template <class M>
class NonHomogeneousParameterMesh : public ParameterMesh<M>
{
private:
	map< int, map<int, ParameterSpace> > _prototypes;
	map< int, map<int, ParameterSpace> > _interpolated;

	int _samplePoints;
	int _meshSize;
	int _delta;

	bool _computed;

	void compute()
	{
		_interpolated.clear();

		for (int param = 0; param < MAX_PARAMS; param++)
		{
			Interpolator interpolator(_meshSize, _samplePoints);
			for (map<int, map<int, ParameterSpace> >::iterator iter = _prototypes.begin(); iter != _prototypes.end(); ++iter)
			{
				int x = iter->first;
				for (map<int, ParameterSpace>::iterator yIter = iter->second.begin(); yIter != iter->second.end(); ++yIter)
				{
					int y = yIter->first;

					interpolator.addPoint(x, y, yIter->second[param]);
				}
			}

			printf("Interpolating %i\n", param);
			double * interpolated = interpolator.interpolateMesh();

			for (int y = 0; y < _meshSize; y++)
			{
				for (int x = 0; x < _meshSize; x++)
				{
					_interpolated[x][y][param] = interpolated[x + y * _samplePoints];
				}
			}

			delete[] interpolated;
		}


		precalculateAll();
	}

	void precalculateAll()
	{
		for (int x = 0; x < _meshSize; x++)
		{
			for (int y = 0; y < _meshSize; y++)
			{
				M::precalculate(_interpolated[x][y]);
			}
		}
	}


public:
	NonHomogeneousParameterMesh(int samplePoints, int meshSize) :
		_samplePoints(samplePoints),
		_meshSize(meshSize),
		_delta(_meshSize / _samplePoints),
		_computed(false)
	{

	}

	void addRefPoint(int x, int y, ParameterSpace & prototype)
	{
		_prototypes[x][y] = prototype;
	}

	virtual ParameterSpace paramsAt(int x, int y)
	{
		if (!_computed)
		{
			compute();
			_computed = true;
		}
		return _interpolated[x][y];
	}

	virtual string describe()
	{
		return "Non-Homogeneous parameters";
	}

	virtual ~NonHomogeneousParameterMesh()
	{
	}
};




#endif /* NONHOMOGENEOUSPARAMETERMESH_CUH_ */
