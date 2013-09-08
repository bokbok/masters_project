/*
 * Partition.cuh
 *
 *  Created on: 07/09/2013
 *      Author: matt
 */

#ifndef PARTITION_CUH_
#define PARTITION_CUH_

#include <string>
#include "../../ParameterSpace.cuh"
#include "../yaml/YAMLParamReader.cuh"

using namespace std;

template <class M>
class Partition
{
private:
	ParameterSpace * _params;
	string _paramFile;
	double _x, _y;

public:
	Partition(string paramFile, double x, double y) :
		_params(NULL),
		_paramFile(paramFile),
		_x(x),
		_y(y)
	{
	}

	virtual ~Partition()
	{
		delete _params;
	}

	double xPos()
	{
		return _x;
	}

	double yPos()
	{
		return _y;
	}

	virtual bool covers(double x, double y) = 0;
	virtual string describe()
	{
		char buf[256];
		sprintf(buf, "Partition located (%i, %i): %s", _x, _y, _paramFile.c_str());
		return buf;
	}

	ParameterSpace vals()
	{
		if (_params == NULL)
		{
			YAMLParamReader<M> reader(_paramFile);
			_params = new ParameterSpace(reader.read());
		}

		return *_params;
	}
};


#endif /* PARTITION_CUH_ */
