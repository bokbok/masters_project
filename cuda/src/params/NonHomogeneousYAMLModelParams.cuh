/*
 * NonHomogeneousYAMLModelParams.cuh
 *
 *  Created on: 04/08/2013
 *      Author: matt
 */

#ifndef NONHOMOGENEOUSYAMLMODELPARAMS_CUH_
#define NONHOMOGENEOUSYAMLMODELPARAMS_CUH_

#include "Params.cuh"
#include "NonHomogeneousParameterMesh.cuh"
#include "yaml/YAMLParams.hpp"
#include <iostream>
#include <vector>

template <class M>
class NonHomogeneousYAMLModelParams : public Params<M>
{
private:
	vector<ParameterSpace> _points;
	vector<string> _filenames;
	int _samplePoints;

	void readPoints()
	{
		map<string, int> paramLookup = paramMap();

		for (vector<string>::iterator iter = _filenames.begin(); iter != _filenames.end(); ++iter)
		{
			YAMLParams yaml(*iter);
			map<string, double> readParams = yaml.read();
			ParameterSpace point;

			cout << *iter << ":" << endl;
			for (map<string, double>::iterator yamlIter = readParams.begin(); yamlIter != readParams.end(); ++yamlIter)
			{
				if (paramLookup.find(yamlIter->first) != paramLookup.end())
				{
					cout << yamlIter->first << "=" << yamlIter->second << endl;
					point[paramLookup[yamlIter->first]] = yamlIter->second;
				}
			}
			_points.push_back(point);
		}
	}

public:
	NonHomogeneousYAMLModelParams(vector<string> filenames, int samplePoints) :
		_filenames(filenames),
		_samplePoints(samplePoints)
	{
		readPoints();
	}

	ParameterMesh<M> * mesh(int meshSize)
	{
		NonHomogeneousParameterMesh<M> * mesh = new NonHomogeneousParameterMesh<M>(_samplePoints, meshSize);

		int pt = 0;
		for (int x = 0; x < _samplePoints; x++)
		{
			for (int y = 0; y < _samplePoints; y++)
			{
				mesh->addRefPoint(x, y, _points[pt]);
				pt++;
				pt %= _points.size();
			}
		}

		return mesh;
	}

	StateSpace initialConditions()
	{
		StateSpace initialConditions(M::NUM_DIMENSIONS);

		initialConditions[M::h_e] = _points[0][M::h_e_rest];
		initialConditions[M::h_i] = _points[0][M::h_i_rest];

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
		sprintf(buf, "%i", _samplePoints);
		string desc = "Non-Homogeneous parameters from files with ";
		desc += string(buf) + " by " + string(buf) + " sample points:";
		for (vector<string>::iterator iter = _filenames.begin(); iter != _filenames.end(); ++iter)
		{
			desc += "\n" + *iter;
		}
		return desc;
	}


};


#endif /* NONHOMOGENEOUSYAMLMODELPARAMS_CUH_ */
