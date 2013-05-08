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
	__device__
	int index(int xOffset, int yOffset)
	{
		int xIdx = fixFor((_x + xOffset), _width);
		int yIdx = fixFor((_y + yOffset), _height);
		return xIdx + yIdx * _width;
	}

	__device__
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
	DeviceMeshPoint(StateSpace * mesh, ParameterSpace * parameters, int width, int height, int x, int y, double delta) :
		_mesh(mesh), _width(width), _height(height), _x(x), _y(y), _delta(delta), _parameters(parameters) { }

	__device__
	StateSpace & stateAt(int xOffset, int yOffset)
	{
		return _mesh[index(xOffset, yOffset)];
	}

	__device__
	StateSpace & state()
	{
		return stateAt(0, 0);
	}

	__device__
	ParameterSpace & parameters()
	{
		return parametersAt(0, 0);
	}


	__device__
	ParameterSpace & parametersAt(int xOffset, int yOffset)
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

	double _delta;
};


#endif /* DEVICEMESH_CUH_ */
