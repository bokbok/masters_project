/*
 * ShapePartitionedParameterMesh.cuh
 *
 *  Created on: 07/09/2013
 *      Author: matt
 */

#ifndef SHAPEPARTITIONEDPARAMETERMESH_CUH_
#define SHAPEPARTITIONEDPARAMETERMESH_CUH_

#include "ParameterMesh.cuh"
#include <vector>
#include "shapes/Partition.cuh"

using namespace std;

template <class T>
class ShapePartitionedParameterMesh : public ParameterMesh<T>
{
private:
	ParameterSpace _base;
	vector<Partition<T> *> _partitions;
	int _meshSize;

	double normalise(int coord)
	{
		return (double) coord / (double) _meshSize;
	}
public:
	ShapePartitionedParameterMesh(ParameterSpace base, vector<Partition<T> *> partitions, int meshSize) :
		_base(base),
		_partitions(partitions),
		_meshSize(meshSize)
	{
	}

	virtual ParameterSpace paramsAt(int x, int y)
	{
		for (int i = 0; i < _partitions.size(); i++)
		{
			if (_partitions[i]->covers(normalise(x), normalise(y)))
			{
				return _partitions[i]->vals();
			}
		}
		printf("\n");

		return _base;
	}
};

#endif /* SHAPEPARTITIONEDPARAMETERMESH_CUH_ */
