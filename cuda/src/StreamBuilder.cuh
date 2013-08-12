/*
 * StreamBuilder.cuh
 *
 *  Created on: 22/06/2013
 *      Author: matt
 */

#ifndef STREAMBUILDER_CUH_
#define STREAMBUILDER_CUH_

#include "io/FileDataStream.cuh"
#include "io/BinaryDataStream.cuh"
#include "io/AsyncDataStream.cuh"
#include "io/CompositeDataStream.cuh"
#include "io/visual/FrameRenderingDataStream.cuh"
#include "io/visual/TraceRenderingDataStream.cuh"
#include "io/monitor/ConvergenceMonitor.cuh"

#include <map>
#include <vector>

#include <ctime>

using namespace std;

class StreamBuilder
{
private:
	vector<DataStream *> _streams;
	vector<DataStream *> _all;

	CompositeDataStream * _built;

	int _meshSize;

	string _runPath;

	void addAsync(DataStream * stream)
	{
		AsyncDataStream * async = new AsyncDataStream(*stream);

		_all.push_back(async);
		_all.push_back(stream);

		_streams.push_back(async);
	}

	string runPath(string basePath)
	{
		double sysTime = time(0);

		char buf[600];
		sprintf(buf, "%i", (int) sysTime);

		mkdir(basePath.c_str(), 0777);
		string runPath = basePath + "/" + buf;
		mkdir(runPath.c_str(), 0777);
		return runPath;
	}


public:
	StreamBuilder(int meshSize, string basePath) :
		_meshSize(meshSize),
		_built(NULL)
	{
		_runPath = runPath(basePath);
	}

	StreamBuilder & toFile(map<string, int> dimensions)
	{
		addAsync(new FileDataStream(_runPath + "/run.dat", dimensions));

		return *this;

	}

	StreamBuilder & toBinaryFile(map<string, int> dimensions)
	{
		addAsync(new BinaryDataStream(_runPath + "/run.dat.bin", dimensions));

		return *this;

	}

	StreamBuilder & RMSFor(int dim, int every, double restVal)
	{
		addAsync(new FrameRenderingDataStream(_runPath, _meshSize, _meshSize, dim, every, restVal));

		return *this;
	}

	StreamBuilder & traceFor(int dim, int every, int spacing, double yMin, double yMax)
	{
		addAsync(new TraceRenderingDataStream(_runPath, _meshSize, _meshSize, dim, _meshSize / spacing, every, yMin, yMax));

		return *this;
	}

	StreamBuilder & monitorConvergence()
	{
		addAsync(new ConvergenceMonitor());

		return *this;
	}

	string runPath()
	{
		return _runPath;
	}

	DataStream * build()
	{
		_built = new CompositeDataStream(_streams);

		return _built;
	}

	~StreamBuilder()
	{
		vector<DataStream *>::iterator iter;

		if (_built != NULL)
		{
			_built->waitToDrain();
			delete _built;
		}

		for (iter = _all.begin(); iter != _all.end(); ++iter)
		{
			delete *iter;
		}
	}
};


#endif /* STREAMBUILDER_CUH_ */
