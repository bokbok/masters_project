/*
 * PartitionedYamlParams.cuh
 *
 *  Created on: 29/08/2013
 *      Author: matt
 */

#ifndef PARTITIONEDYAMLPARAMS_CUH_
#define PARTITIONEDYAMLPARAMS_CUH_

#include "Params.cuh"
#include <vector>
#include <string>
#include "yaml/YAMLParamReader.cuh"
#include "PartitionedParameterMesh.cuh"

using namespace std;

template <class M>
class PartitionedYAMLParams : public Params<M>
{
private:
	vector<string> _filenames;
	ParameterMesh<M> * _mesh;
	vector<ParameterSpace> _regions;

	void loadRegions()
	{
		for (vector<string>::iterator iter = _filenames.begin(); iter != _filenames.end(); ++iter)
		{
			YAMLParamReader<M> reader(*iter);
			_regions.push_back(reader.read());
		}
	}

public:
	PartitionedYAMLParams(vector<string> filenames) :
		_filenames(filenames)
	{
		loadRegions();
	}

	~PartitionedYAMLParams()
	{

	}

	virtual ParameterMesh<M> * mesh(int meshSize)
	{
		return new PartitionedParameterMesh<M>(_regions, meshSize);
	}

	StateSpace initialConditions()
	{
		StateSpace initialConditions(M::NUM_DIMENSIONS);

		initialConditions[M::h_e] = _regions[0][M::h_e_rest];
		initialConditions[M::h_i] = _regions[0][M::h_i_rest];

		initialConditions[M::C_e] = 1;
		initialConditions[M::C_i] = 1;

		return initialConditions;
	}

	map<string, int> stateMap()
	{
		return M::stateMap();
	}

	map<string, int> paramMap()
	{
		return M::paramMap();
	}

	virtual string describe()
	{
		char buf[16];
		sprintf(buf, "%i", _regions.size());
		string desc = "Partitioned parameters from files with ";
		desc += string(buf) + " regions:";
		for (vector<string>::iterator iter = _filenames.begin(); iter != _filenames.end(); ++iter)
		{
			desc += "\n" + *iter;
		}
		return desc;
	}

};


#endif /* PARTITIONEDYAMLPARAMS_CUH_ */
