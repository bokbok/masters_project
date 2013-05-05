/*
 * Mesh.cuh
 *
 *  Created on: 29/04/2013
 *      Author: matt
 */

#ifndef MESH_CUH_
#define MESH_CUH_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <vector>

#include "DeviceMeshPoint.cuh"
#include "RungeKuttaIntegrator.cuh"
#include "liley/Model.cuh"

using namespace std;


__global__
void __callStep(StateSpace * prev, StateSpace * curr, ParameterSpace * parameters, int width, int height, double delta, double t, double deltaT)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;


	DeviceMeshPoint current(curr, *parameters, width, height, x, y, delta);
	DeviceMeshPoint previous(prev, *parameters, width, height, x, y, delta);

	RungeKuttaIntegrator<Model> integrator(current, previous, t, deltaT);
	integrator.integrateStep();
}

class Mesh
{
public:
	Mesh(int width, int height, double delta, int flushSteps):
		_width(width),
		_height(height),
		_delta(delta),
		_flushSteps(flushSteps)
	{
		allocate();
	}

	virtual ~Mesh()
	{
		deallocate();
	}

	void stepAndFlush(double t, double deltaT, ostream & out)
	{
		cudaStep(t, deltaT);
		_sheet++;

		if (_sheet > _flushSteps)
		{
			flush(out);
			_sheet = 0;
		}
	}

private:
	int _width, _height;
	int _flushSteps;
	double _delta;

	vector< thrust::device_vector<StateSpace> > _sheets;
	vector< thrust::device_ptr<StateSpace> > _pointers;
	thrust::device_ptr<ParameterSpace> _parameters;

	int _sheet;

	void flush(ostream & out)
	{
		// output
	}

	void cudaStep(double t, double deltaT)
	{
		thrust::device_vector<StateSpace> & current = _sheets[_sheet];
		thrust::device_vector<StateSpace> & prev = _sheets[(_sheet - 1) % _sheets.size()];

		thrust::device_ptr<StateSpace> & currentPtr = _pointers[_sheet];
		thrust::device_ptr<StateSpace> & prevPtr = _pointers[(_sheet - 1) % _pointers.size()];

		dim3 grid(_width / 10, _height / 10), block(10, 10);

		printf("\nb4 kernel %p %p", (StateSpace *)thrust::raw_pointer_cast( prevPtr ), (StateSpace *)thrust::raw_pointer_cast( currentPtr ));
		__callStep<<< grid, block >>>((StateSpace *)thrust::raw_pointer_cast( prevPtr ), (StateSpace *)thrust::raw_pointer_cast( currentPtr ), (ParameterSpace *)thrust::raw_pointer_cast( _parameters ), _width, _height, _delta, t, deltaT);
		//cudaDeviceSynchronize();
	}


	void allocate()
	{
		_sheet = 0;

		for (int sheet = 0; sheet < _flushSteps; sheet++)
		{
			int N = _width * _height;
			thrust::device_ptr<StateSpace> mem = thrust::device_new<StateSpace>(thrust::device_new<StateSpace>(N), StateSpace(), N);

			thrust::device_vector<StateSpace> v(mem, mem + N);
			_sheets.push_back(v);
			_pointers.push_back(mem);
		}
		_parameters = thrust::device_new<ParameterSpace>(thrust::device_new<ParameterSpace>(), ParameterSpace());
	}

	void deallocate()
	{
		for (int i = 0; i < _pointers.size(); i++)
		{
			thrust::device_free(_pointers[i]);
		}
		thrust::device_free(_parameters);
	}
};


#endif /* MESH_CUH_ */
