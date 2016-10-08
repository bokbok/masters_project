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
	Buffer * _data;
	int _width, _height;

public:
	Write(Buffer * data) :
		_data(data)
	{
		_data->checkOut();
	}

	void to(DataStream & stream)
	{
		stream.write(_data);
	}

	~Write()
	{
		_data->release();
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
	volatile bool _empty;

	void startWriteThread()
	{
		pthread_mutex_init(&_dataMutex, NULL);
		pthread_cond_init(&_dataAvailable, NULL);

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
		_empty = true;
		pthread_cond_wait(&_dataAvailable, &_dataMutex);
		_empty = false;
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
				try
				{
					write->to(_decorated);
				}
				catch (...)
				{
					printf("Unhandled exception!");
				}
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

	virtual void write(Buffer * data)
	{
		Write * write = new Write(data);
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
			timespec ts;
			ts.tv_sec = 0;
			ts.tv_nsec = 10000;
			nanosleep(&ts, NULL);
		}
	}

	bool empty()
	{
		pthread_mutex_lock(&_dataMutex);
		bool result = _empty;
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
		cleanup();
	}

	void cleanup()
	{
		Write * write;
		while((write = nextWrite()) != NULL)
		{
			delete write;
		}
	}

};


#endif /* ASYNCDATASTREAM_CUH_ */
