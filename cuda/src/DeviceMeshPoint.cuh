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
public:
	__device__
	DeviceMeshPoint(StateSpace * mesh, ParameterSpace & parameters, int width, int height, int x, int y, double delta) :
		_mesh(mesh), _width(width), _height(height), _x(x), _y(y), _delta(delta), _parameters(parameters)
	{

	}

	__device__
	StateSpace & stateAt(int xOffset, int yOffset)
	{
		return _mesh[((_x + xOffset) % _width) + ((_y + yOffset) % _height) * _width];
	}

	__device__
	StateSpace & state()
	{
		return stateAt(0, 0);
	}

	__device__
	ParameterSpace & parameters()
	{
		return _parameters;
	}


private:
	StateSpace * _mesh;
	ParameterSpace & _parameters;
	int _width, _height;
	int _x, _y;

	double _delta;
};


#endif /* DEVICEMESH_CUH_ */
