/*
 * CircularPartition.cuh
 *
 *  Created on: 07/09/2013
 *      Author: matt
 */

#ifndef CIRCULARPARTITION_CUH_
#define CIRCULARPARTITION_CUH_

#include "Partition.cuh"
#include <string>

template <class T>
class CircularPartition : public Partition<T>
{
private:
	double _r, _x, _y;

public:
	CircularPartition(string paramFile, double x, double y, double r) : Partition<T>(paramFile, x, y), _r(r), _x(y), _y(y)
	{

	}

	virtual ~CircularPartition()
	{
	}

	virtual bool covers(double x, double y)
	{
		double dX = _x - x;
		double dY = _y - y;

		double radius = sqrt(dX * dX + dY * dY);

		return radius < _r;
	}
};


#endif /* CIRCULARPARTITION_CUH_ */
