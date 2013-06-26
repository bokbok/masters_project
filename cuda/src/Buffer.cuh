/*
 * Buffer.cuh
 *
 *  Created on: 10/06/2013
 *      Author: matt
 */

#ifndef BUFFER_CUH_
#define BUFFER_CUH_
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <pthread.h>

#include "StateSpace.cuh"


class Buffer
{

private:
	volatile int _refCount;
	thrust::host_vector<StateSpace> _transferBuffer;
	//thrust::device_ptr<StateSpace> & _src;
	int _N, _width, _height;

	pthread_mutex_t _dataMutex;


public:
	Buffer(thrust::device_ptr<StateSpace> src, int width, int height, int N) :
		_width(width),
		_height(height),
		_N(N),
		_transferBuffer(N),
		_refCount(0)
	{
		pthread_mutex_init(&_dataMutex, NULL);

		thrust::device_vector<StateSpace> device(src, src + _N);
		thrust::copy(device.begin(), device.end(), _transferBuffer.begin());
	}

	void checkOut()
	{
		pthread_mutex_lock(&_dataMutex);
		++_refCount;
		pthread_mutex_unlock(&_dataMutex);
	}

	void release()
	{
		pthread_mutex_lock(&_dataMutex);
		--_refCount;
		if(_refCount == 0)
		{
			delete this;
		}
		else
		{
			pthread_mutex_unlock(&_dataMutex);
		}
	}

	StateSpace & operator [](int index)
	{
		return data()[index];
	}

	operator StateSpace *()
	{
		return data();
	}

	StateSpace * data()
	{
		return _transferBuffer.data();
	}

	int width()
	{
		return _width;
	}

	int height()
	{
		return _height;
	}

	int length()
	{
		return _N;
	}

	~Buffer()
	{
		pthread_mutex_unlock(&_dataMutex);
	}
};


#endif /* BUFFER_CUH_ */
