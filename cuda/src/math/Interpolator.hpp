/*
 * Interpolator.hpp
 *
 *  Created on: 27/07/2013
 *      Author: matt
 */

#ifndef INTERPOLATOR_HPP_
#define INTERPOLATOR_HPP_
#include <vector>

using namespace std;

class Interpolator
{
	int _meshSize, _samplePoints;
	double * _points;

public:
	Interpolator(int meshSize, int samplePoints) :
		_meshSize(meshSize),
		_samplePoints(samplePoints)
	{
		_points = new double[_samplePoints * _samplePoints];
	}

	~Interpolator()
	{
		delete[] _points;
	}

	void addPoint(int x, int y, double val)
	{
		_points[x + y * _samplePoints] = val;
	}

	double * interpolateMesh();

};


#endif /* INTERPOLATOR_HPP_ */
