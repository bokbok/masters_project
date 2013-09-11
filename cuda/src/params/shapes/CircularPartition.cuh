/*
 * CircularPartition.cuh
 *
 *  Created on: 07/09/2013
 *      Author: matt
 */

#ifndef CIRCULARPARTITION_CUH_
#define CIRCULARPARTITION_CUH_

#include "Partition.cuh"
#include "../../math/BoundaryInterpolator.cuh"
#include <string>

template <class T>
class CircularPartition : public Partition<T>
{
private:
	double _r, _x, _y, _boundaryRadius;
	string _paramFile;
	ParameterSpace * _params;

	double radius(double x, double y)
	{
		double dX = _x - x;
		double dY = _y - y;

		return sqrt(dX * dX + dY * dY);
	}

public:
	CircularPartition(string paramFile, double x, double y, double r, double boundary) :
		_r(r),
		_x(y),
		_y(y),
		_boundaryRadius(r * (1 + boundary)),
		_paramFile(paramFile),
		_params(NULL)
	{
	}

	virtual ~CircularPartition()
	{
		delete _params;
	}

	virtual bool covers(double x, double y)
	{
		return radius(x, y) < _boundaryRadius;
	}

	virtual string describe()
	{
		char buf[1024];
		sprintf(buf, "Circular buffer centered at (%f, %f) radius %f (NORMALISED). Parameters from %s", _x, _y, _r, _paramFile.c_str());

		return buf;
	}

	virtual ParameterSpace vals(double x, double y, ParameterSpace & base)
	{
		if (_params == NULL)
		{
			YAMLParamReader<T> reader(_paramFile);
			_params = new ParameterSpace(reader.read());
		}

		double dist = radius(x, y);
		if (dist < _r)
		{
			return *_params;
		}
		else
		{
			double delta = _boundaryRadius - dist;

			BoundaryInterpolator interpolator(_boundaryRadius - _r, base, *_params);

			return interpolator.interpolate(delta);
		}
	}

};


#endif /* CIRCULARPARTITION_CUH_ */
