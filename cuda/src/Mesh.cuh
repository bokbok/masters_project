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

using namespace std;

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
