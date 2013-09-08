/*
 * ShapePartitionedParams.cuh
 *
 *  Created on: 07/09/2013
 *      Author: matt
 */

#ifndef SHAPEPARTITIONEDPARAMS_CUH_
#define SHAPEPARTITIONEDPARAMS_CUH_

#include "Params.cuh"
#include "yaml/YAMLParamReader.cuh"
#include "shapes/Partition.cuh"
#include "ShapePartitionedParameterMesh.cuh"
#include <vector>

template <class T>
class ShapePartitionedParams : public Params<T>
{
private:
	string _baseFile;
	vector<Partition<T> *> _partitions;
	ParameterSpace _base;


public:
	ShapePartitionedParams(string baseFile) :
		_baseFile(baseFile)
	{
		YAMLParamReader<T> reader(baseFile);
		_base = reader.read();
	}

	virtual ~ShapePartitionedParams()
	{
	}

	ShapePartitionedParams & addPartition(Partition<T> * partition)
	{
		_partitions.push_back(partition);
		return *this;
	}

	virtual ParameterMesh<T> * mesh(int meshSize)
	{
		return new ShapePartitionedParameterMesh<T>(_base, _partitions, meshSize);
	}

	virtual StateSpace initialConditions()
	{
		StateSpace initialConditions(T::NUM_DIMENSIONS);

		initialConditions[T::h_e] = _base[T::h_e_rest];
		initialConditions[T::h_i] = _base[T::h_i_rest];

		initialConditions[T::C_e] = 1;
		initialConditions[T::C_i] = 1;

		return initialConditions;
	}

	virtual map<string, int> paramMap()
	{
		return T::paramMap();
	}

	virtual map<string, int> stateMap()
	{
		return T::stateMap();
	}

	virtual string describe()
	{
		string desc = "";
		for (int i = 0; i < _partitions.size(); i++)
		{
			desc += _partitions[i]->describe();
		}

		return desc;
	}
};


#endif /* SHAPEPARTITIONEDPARAMS_CUH_ */
