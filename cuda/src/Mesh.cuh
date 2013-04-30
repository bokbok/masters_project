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

using namespace std;


template<class T>
__global__
void __call(T * prev, T * curr, int width, int height, double delta)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;


	//printf("In call! %i %i", width, height);
	//printf("In call! %i %i", x, y);
	//printf("\nIn call! %p %p", prev, curr);
	//printf("\nMesh point %p %p", &prev[x + y * width], &curr[x + y * width]);

	DeviceMeshPoint<T> current(curr, width, height, x, y, delta);
	DeviceMeshPoint<T> previous(prev, width, height, x, y, delta);

	// Integrator.integrate(current, previous, t)
}

template<class T> class Mesh
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

	void stepAndFlush(ostream & out)
	{
		cudaStep();
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

	vector< thrust::device_vector<T> > _sheets;
	vector< thrust::device_ptr<T> > _pointers;

	int _sheet;

	void flush(ostream & out)
	{
		// output
	}

	void cudaStep()
	{
		thrust::device_vector<T> & current = _sheets[_sheet];
		thrust::device_vector<T> & prev = _sheets[(_sheet - 1) % _sheets.size()];

		thrust::device_ptr<T> & currentPtr = _pointers[_sheet];
		thrust::device_ptr<T> & prevPtr = _pointers[(_sheet - 1) % _pointers.size()];

		dim3 grid(_width / 10, _height / 10), block(10, 10);

		printf("\nb4 kernel %p %p", (T *)thrust::raw_pointer_cast( prevPtr ), (T *)thrust::raw_pointer_cast( currentPtr ));
		__call<<< grid, block >>>((T *)thrust::raw_pointer_cast( prevPtr ), (T *)thrust::raw_pointer_cast( currentPtr ), _width, _height, _delta);
		cudaDeviceSynchronize();
	}


	void allocate()
	{
		_sheet = 0;

		for (int sheet = 0; sheet < _flushSteps; sheet++)
		{
			int N = _width * _height;
			thrust::device_ptr<T> mem = thrust::device_new<T>(thrust::device_new<T>(N), T::zero, N);

			thrust::device_vector<T> v(mem, mem + N);
			_sheets.push_back(v);
			_pointers.push_back(mem);
		}
	}

	void deallocate()
	{
		for (int i = 0; i < _pointers.size(); i++)
		{
			thrust::device_free(_pointers[i]);
		}
	}
};


#endif /* MESH_CUH_ */
