/*
 * YAMLParamReader.cuh
 *
 *  Created on: 29/08/2013
 *      Author: matt
 */

#ifndef YAMLPARAMREADER_CUH_
#define YAMLPARAMREADER_CUH_

#include "YAMLParams.hpp"
#include <string>
#include "../../ParameterSpace.cuh"

template <class M>
class YAMLParamReader
{
private:
	string _filename;
public:
	YAMLParamReader(string filename):
		_filename(filename)
	{
	}

	ParameterSpace read()
	{
		YAMLParams yaml(_filename);
		map<string, double> readParams = yaml.read();
		ParameterSpace point;

		map<string, int> paramLookup = M::paramMap();

		cout << _filename << ":" << endl;
		for (map<string, double>::iterator yamlIter = readParams.begin(); yamlIter != readParams.end(); ++yamlIter)
		{
			if (paramLookup.find(yamlIter->first) != paramLookup.end())
			{
				cout << yamlIter->first << "=" << yamlIter->second << endl;
				point[paramLookup[yamlIter->first]] = yamlIter->second;
			}
		}

		return point;
	}
};


#endif /* YAMLPARAMREADER_CUH_ */
