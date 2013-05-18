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
#include "io/DataStream.cuh"

using namespace std;


template <class M>
__global__
void __callStep(StateSpace * prev, StateSpace * curr, ParameterSpace * parameters, int width, int height, double delta, double t, double deltaT, M model)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	DeviceMeshPoint current(curr, parameters, width, height, x, y, delta);
	DeviceMeshPoint previous(prev, parameters, width, height, x, y, delta);

	RungeKuttaIntegrator<M> integrator(current, previous, t, deltaT, delta);
	integrator.integrateStep();
}

const int BLOCK_SIZE = 10;
template <class M>
class Mesh
{
public:
	Mesh(int width, int height, double delta, int flushSteps, StateSpace initialConditions, ParameterSpace params):
		_width(width),
		_height(height),
		_delta(delta),
		_flushSteps(flushSteps),
		_initialConditions(initialConditions),
		_params(params),
		_N(width * height)
	{
		allocate();
	}

	virtual ~Mesh()
	{
		deallocate();
	}

	void stepAndFlush(double t, double deltaT, DataStream & out)
	{
		cudaStep(t, deltaT);
		_sheet++;

		if (_sheet == _flushSteps)
		{
			flush(out);
			_sheet = 0;
		}
	}

private:
	int _width, _height, _N;
	int _flushSteps;
	StateSpace & _initialConditions;
	ParameterSpace & _params;
	double _delta;

	vector< thrust::device_ptr<StateSpace> > _pointers;
	thrust::device_ptr<ParameterSpace> _parameters;

	vector< thrust::host_vector <StateSpace> * > _hostBuffers;

	int _sheet;

	void flush(DataStream & out)
	{
		cudaDeviceSynchronize();
		printf("Flushing ...\n");

		out.waitToDrain();
		printf("Stream ready ...\n");

		for (int i = 0; i < _flushSteps; i++)
		{
			thrust::host_vector <StateSpace> * buffer = _hostBuffers[i];
			thrust::device_vector<StateSpace> device(_pointers[i], _pointers[i] + _N);

			thrust::copy(device.begin(), device.end(), buffer->begin());
			out.write(buffer->data(), _width, _height);
		}

		printf("Flushed ...\n");
	}

	void cudaStep(double t, double deltaT)
	{
		int prevIndex = _sheet - 1;

		if (prevIndex < 0)
		{
			prevIndex = _flushSteps - 1;
		}

		thrust::device_ptr<StateSpace> & currentPtr = _pointers[_sheet];
		thrust::device_ptr<StateSpace> & prevPtr = _pointers[prevIndex];

		dim3 grid(_width / BLOCK_SIZE, _height / BLOCK_SIZE), block(BLOCK_SIZE, BLOCK_SIZE);

		__callStep<<< grid, block >>>((StateSpace *)thrust::raw_pointer_cast( prevPtr ),
									  (StateSpace *)thrust::raw_pointer_cast( currentPtr ),
									  (ParameterSpace *)thrust::raw_pointer_cast( _parameters ),
									  _width, _height, _delta, t, deltaT, M());
	}


	void allocate()
	{
		_sheet = 0;

		for (int sheet = 0; sheet < _flushSteps; sheet++)
		{
			thrust::device_ptr<StateSpace> mem = thrust::device_new<StateSpace>(thrust::device_new<StateSpace>(_N), _initialConditions, _N);
			thrust::host_vector <StateSpace> * buffer = new thrust::host_vector <StateSpace>(_N);

			_pointers.push_back(mem);
			_hostBuffers.push_back(buffer);
		}
		_parameters = thrust::device_new<ParameterSpace>(thrust::device_new<ParameterSpace>(_N), _params, _N);
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
