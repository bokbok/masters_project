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
#include "params/ParameterMesh.cuh"
#include "common.cuh"
#include "Buffer.cuh"
#include <time.h>


using namespace std;

template <class M>
__global__
void __callStep(StateSpace * prev,
					   StateSpace * curr,
					   ParameterSpace * parameters,
					   int width,
					   int height,
					   double delta,
					   double t,
					   double deltaT)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	DeviceMeshPoint current(curr, parameters, width, height, x, y, delta);
	DeviceMeshPoint previous(prev, parameters, width, height, x, y, delta);

	RungeKuttaIntegrator<M> integrator(current, previous, t, deltaT);
	integrator.integrateStep();
}


template <class M>
class Mesh
{
public:
	Mesh(int width, int height, double delta, int bufferSize, int reportSteps, StateSpace * initialConditions, ParameterMesh<M> & params):
		_width(width),
		_height(height),
		_delta(delta),
		_bufferSize(bufferSize),
		_stepNum(0),
		_reportSteps(reportSteps),
		_initialConditions(initialConditions),
		_params(params),
		_N(width * height),
		_flushCount(0)
	{
		allocate();
		cudaFuncSetCacheConfig(__callStep<M>, cudaFuncCachePreferL1);
	}

	virtual ~Mesh()
	{
		deallocate();
	}

	void stepAndFlush(double t, double deltaT, DataStream & out)
	{
		cudaStep(t, deltaT);

		if ((_stepNum % _reportSteps) == 0)
		{
			flush(out);
		}

		_stepNum++;
	}

private:
	int _width, _height, _N;
	int _bufferSize, _reportSteps, _stepNum;
	StateSpace * _initialConditions;
	ParameterMesh<M> & _params;
	double _delta;

	int _flushCount;

	vector< thrust::device_ptr<StateSpace> > _pointers;
	thrust::device_ptr<ParameterSpace> _parameters;

	void flush(DataStream & out)
	{
		int sheetToWrite = _stepNum % _bufferSize;

		clock_t start = clock();
		cudaDeviceSynchronize();

		clock_t transferStart = clock();
		Buffer * buffer = new Buffer(_pointers[sheetToWrite], _width, _height, _N);
		buffer->checkOut();
		out.write(buffer);
		clock_t finish = clock();
		_flushCount++;
		printf("Flushed(%i)(time taken:%i ms, transfer time:%i ms) t=%f\n", _flushCount, (finish - start) / 1000, (finish - transferStart) / 1000,buffer->data()[0].t());
		buffer->release();
	}

	void cudaStep(double t, double deltaT)
	{
		int sheet = _stepNum % _bufferSize;

		int prevIndex = sheet - 1;

		if (prevIndex < 0)
		{
			prevIndex = _bufferSize - 1;
		}

		thrust::device_ptr<StateSpace> & currentPtr = _pointers[sheet];
		thrust::device_ptr<StateSpace> & prevPtr = _pointers[prevIndex];

		dim3 grid(_width / BLOCK_SIZE, _height / BLOCK_SIZE), block(BLOCK_SIZE, BLOCK_SIZE);

		__callStep<M><<< grid, block >>>((StateSpace *)thrust::raw_pointer_cast( prevPtr ),
									  (StateSpace *)thrust::raw_pointer_cast( currentPtr ),
									  (ParameterSpace *)thrust::raw_pointer_cast( _parameters ),
									  _width, _height, _delta, t, deltaT);
	}


	void allocate()
	{
		thrust::host_vector<StateSpace> ics(_initialConditions, _initialConditions + _N);
		for (int sheet = 0; sheet < _bufferSize; sheet++)
		{
			thrust::device_ptr<StateSpace> mem = thrust::device_new<StateSpace>(_N);
			thrust::copy(ics.begin(), ics.end(), mem);
			_pointers.push_back(mem);
		}

		_parameters = thrust::device_new<ParameterSpace>(_N);
		vector<ParameterSpace> params;
		for (int x = 0; x < _width; x++)
		{
			for (int y = 0; y < _height; y++)
			{
				params.push_back(_params.paramsAt(x, y));
			}
		}
		thrust::copy(params.begin(), params.end(), _parameters);
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
