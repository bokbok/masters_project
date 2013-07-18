/*
 * DeviceMesh.cuh
 *
 *  Created on: 30/04/2013
 *      Author: matt
 */

#ifndef DEVICEMESH_CUH_
#define DEVICEMESH_CUH_

#include "StateSpace.cuh"


class DeviceMeshPoint
{
private:
	inline __device__
	int index(int xOffset, int yOffset)
	{
		int xIdx = fixFor((_x + xOffset), _width);
		int yIdx = fixFor((_y + yOffset), _height);
		return xIdx + yIdx * _width;
	}

	inline __device__
	int fixFor(int idx, int max)
	{
		if (idx < 0)
		{
			idx += max;
		}
		else if (idx >= max)
		{
			idx -= max;
		}
		return idx;
	}


public:
	__device__
	inline DeviceMeshPoint(StateSpace * mesh, ParameterSpace * parameters, int width, int height, int x, int y, double delta) :
		_mesh(mesh), _width(width), _height(height), _x(x), _y(y), _delta(delta), _parameters(parameters), _delta2(delta * delta), _denominator(delta * delta * 180)
	{
	}

	__device__
	inline StateSpace & stateAt(int xOffset, int yOffset)
	{
		return _mesh[index(xOffset, yOffset)];
	}

	__device__
	inline StateSpace & state()
	{
		return stateAt(0, 0);
	}

	__device__
	inline ParameterSpace & parameters()
	{
		return parametersAt(0, 0);
	}

	__device__
	inline double d2dx2(int dim)
	{
//		double xMinus = stateAt(-1, 0)[dim];
//		double x = state()[dim];
//		double xPlus = stateAt(1, 0)[dim];
//
//		return (xMinus - 2 * x + xPlus) / _delta2;

		double xMinus = stateAt(-1, 0)[dim];
		double xMinus2 = stateAt(-2, 0)[dim];
		double xMinus3 = stateAt(-3, 0)[dim];
		double x = state()[dim];
		double xPlus = stateAt(1, 0)[dim];
		double xPlus2 = stateAt(2, 0)[dim];
		double xPlus3 = stateAt(3, 0)[dim];

		return (2 * (xMinus3 + xPlus3) - 27 * (xPlus2 + xMinus2) + 270 * (xMinus + xPlus) - 490 * x) / _denominator;
	}

	__device__
	inline double d2dy2(int dim)
	{
//		double yMinus = stateAt(0, -1)[dim];
//		double y = state()[dim];
//		double yPlus = stateAt(0, 1)[dim];
//
//		return (yMinus - 2 * y + yPlus) / _delta2;
		double yMinus = stateAt(0, -1)[dim];
		double yMinus2 = stateAt(0, -2)[dim];
		double yMinus3 = stateAt(0, -3)[dim];
		double y = state()[dim];
		double yPlus = stateAt(0, 1)[dim];
		double yPlus2 = stateAt(0, 2)[dim];
		double yPlus3 = stateAt(0, 3)[dim];

		return (2 * (yMinus3 + yPlus3) - 27 * (yPlus2 + yMinus2) + 270 * (yMinus + yPlus) - 490 * y) / _denominator;
	}

	__device__
	inline double laplacian(int dim)
	{
		return d2dy2(dim) + d2dx2(dim);
	}

	__device__
	inline ParameterSpace & parametersAt(int xOffset, int yOffset)
	{
		return _parameters[index(xOffset, yOffset)];
	}

	__device__
	int x()
	{
		return _x;
	}

	__device__
	int y()
	{
		return _y;
	}

private:
	StateSpace * _mesh;
	ParameterSpace * _parameters;
	int _width, _height;
	int _x, _y;

	double _delta, _delta2, _denominator;
};


#endif /* DEVICEMESH_CUH_ */
