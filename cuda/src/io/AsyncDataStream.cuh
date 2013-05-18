/*
 * AsyncDataStream.cuh
 *
 *  Created on: 18/05/2013
 *      Author: matt
 */

#ifndef ASYNCDATASTREAM_CUH_
#define ASYNCDATASTREAM_CUH_

#include <pthread.h>
#include <queue>

class Write
{
private:
	StateSpace * _data;
	int _width, _height;

public:
	Write(StateSpace * data, int width, int height) :
		_data(data), _width(width), _height(height)
	{

	}

	void to(DataStream & stream)
	{
		stream.write(_data, _width, _height);
	}
};

class AsyncDataStream : public DataStream
{
private:
	DataStream & _decorated;
	pthread_t _thread;
	pthread_cond_t _dataAvailable;
	pthread_mutex_t _dataMutex;
	std::queue<Write *> _queue;

	volatile bool _exit;

	void startWriteThread()
	{
		pthread_mutex_init(&_dataMutex, NULL);
		pthread_cond_init(&_dataAvailable, NULL);
		printf("Waiting starting thread...\n");

		pthread_create(&_thread, NULL, AsyncDataStream::start, this);
	}

	void queue(Write * write)
	{
		pthread_mutex_lock(&_dataMutex);
		_queue.push(write);
		pthread_mutex_unlock(&_dataMutex);
		pthread_cond_signal(&_dataAvailable);
	}

	Write * nextWrite()
	{
		pthread_mutex_lock(&_dataMutex);
		Write * result = NULL;
		if (!_queue.empty())
		{
			result = _queue.front();
			_queue.pop();
		}
		pthread_mutex_unlock(&_dataMutex);

		return result;
	}

	void waitForData()
	{
		pthread_mutex_lock(&_dataMutex);
		printf("Got lock\n");
		printf("Waiting for data to write\n");
		pthread_cond_wait(&_dataAvailable, &_dataMutex);
		printf("Got data to write\n");
		pthread_mutex_unlock(&_dataMutex);
	}

	void run()
	{
		_exit = false;

		for(;;)
		{
			Write * write = nextWrite();
			if (write)
			{
				write->to(_decorated);
				delete write;
			}
			else
			{
				waitForData();
			}

			if (_exit)
			{
				pthread_exit(0);
			}
		}
	}

public:
	AsyncDataStream(DataStream & decorated) : _decorated(decorated)
	{
		startWriteThread();
	}

	static void * start(void * instance)
	{
		AsyncDataStream * stream = (AsyncDataStream *) instance;
		stream->run();

		return NULL;
	}

	virtual void write(StateSpace * data, int width, int height)
	{
		Write * write = new Write(data, width, height);
		queue(write);
	}

	virtual void waitToDrain()
	{
		if (!empty())
		{
			std::cout << "Waiting for stream to drain...." << _queue.size() << std::endl;
		}
		while (!empty())
		{
			usleep(10);
		}
	}

	bool empty()
	{
		pthread_mutex_lock(&_dataMutex);
		bool result = _queue.empty();
		pthread_mutex_unlock(&_dataMutex);

		return result;
	}

	void stop()
	{
		_exit = true;
	}

	virtual ~AsyncDataStream()
	{
		stop();
	}

};


#endif /* ASYNCDATASTREAM_CUH_ */
