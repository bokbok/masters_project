/*
 * TraceRenderingDataStream.cuh
 *
 *  Created on: 30/05/2013
 *      Author: matt
 */

#ifndef TRACERENDERINGDATASTREAM_CUH_
#define TRACERENDERINGDATASTREAM_CUH_

#include "../DataStream.cuh"
#include "../../visual/Trace.cuh"
#include <string>
#include <list>

#include <mgl2/mgl.h>
#include <sys/stat.h>
#include <sys/types.h>


using namespace std;
const int MAX_BUFFER = 2048;

class TraceRenderingDataStream : public DataStream
{
private:
	string _outputPath, _fullOutputPath;
	int _dimensionToPlot;
	int _outputSteps;
	double _deltaT;

	double _minY,_maxY;

	int _currentStep;
	int _fileCount;

	Trace _trace;

	void createDirs()
	{
		mkdir(_outputPath.c_str(), 0777);
		_fullOutputPath = _outputPath + "/trace";
		mkdir(_fullOutputPath.c_str(), 0777);
		char buf[32];
		sprintf(buf, "/%i", _dimensionToPlot);
		_fullOutputPath += buf;
		mkdir(_fullOutputPath.c_str(), 0777);
	}


public:
	TraceRenderingDataStream(string outputPath, int width,
							int height, int dimensionToPlot,
							int outputInterval, int outputSteps,
							double minY,
							double maxY):
		_outputPath(outputPath),
		_currentStep(0),
		_fileCount(1),
		_dimensionToPlot(dimensionToPlot),
		_outputSteps(outputSteps),
		_trace(width, height, dimensionToPlot, outputInterval, minY, maxY)
	{
		createDirs();
	}

	virtual void write(Buffer * data)
	{
		data->checkOut();
		_trace.buffer(data->data());
		_currentStep++;

		if (_currentStep % _outputSteps == 0)
		{
			printf("Writing trace frame....\n");
			char buf[64];
			sprintf(buf, "%05d", _fileCount);
			string filename = _fullOutputPath + "/trace_" + buf;

			_trace.renderFile(filename);
			_fileCount++;
			printf("Writing trace frame done....\n");
		}

		data->release();
	}

	virtual void waitToDrain() { }

	virtual ~TraceRenderingDataStream() {};

};


#endif /* TRACERENDERINGDATASTREAM_CUH_ */
