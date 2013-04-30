/*
 * DeviceMesh.cuh
 *
 *  Created on: 30/04/2013
 *      Author: matt
 */

#ifndef DEVICEMESH_CUH_
#define DEVICEMESH_CUH_


template<class T>
class DeviceMeshPoint
{
public:
	__device__
	DeviceMeshPoint(T * mesh, int width, int height, int x, int y, double delta) :
		_mesh(mesh), _width(width), _height(height), _x(x), _y(y), _delta(delta)
	{

	}

	__device__
	inline T * stateAt(int xOffset, int yOffset)
	{
		return &_mesh[((_x + xOffset) % _width) + ((_y + yOffset) % _height) * _width];
	}

	__device__
	inline T * state()
	{
		return stateAt(0, 0);
	}


private:
	T * _mesh;
	int _width, _height;
	int _x, _y;

	double _delta;
};


#endif /* DEVICEMESH_CUH_ */
