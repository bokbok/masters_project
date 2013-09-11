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
public:
	Partition()
	{
	}

	virtual ~Partition()
	{
	}

	virtual bool covers(double x, double y) = 0;
	virtual string describe() = 0;
	virtual ParameterSpace vals(double x, double y, ParameterSpace & base) = 0;
};


#endif /* PARTITION_CUH_ */
