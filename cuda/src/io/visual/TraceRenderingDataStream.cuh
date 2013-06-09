/*
 * TraceRenderingDataStream.cuh
 *
 *  Created on: 30/05/2013
 *      Author: matt
 */

#ifndef TRACERENDERINGDATASTREAM_CUH_
#define TRACERENDERINGDATASTREAM_CUH_

#include "../DataStream.cuh"
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
	int _width, _height;
	int _dimensionToPlot;
	int _outputInterval, _outputSteps;
	double _deltaT;

	double _minY,_maxY;

	int _currentStep;
	int _fileCount;
	map< string, list<double> > _traces;

	void allocateBuffer()
	{
		for (int x = 0; x < _width; x+= _outputInterval)
		{
			for (int y = 0; y < _height; y+= _outputInterval)
			{
				_traces[traceKey(x, y)] = list<double>();
			}
		}
	}

	string traceKey(int x, int y)
	{
		char buf[100];

		sprintf(buf, "(%i, %i)", x, y);

		return buf;
	}

	void buffer(StateSpace * data)
	{
		for (int x = 0; x < _width; x+= _outputInterval)
		{
			for (int y = 0; y < _height; y+= _outputInterval)
			{
				string key = traceKey(x, y);
				_traces[key].push_back(data[x + _width * y][_dimensionToPlot]);
				if (_traces[key].size() > MAX_BUFFER)
				{
					_traces[key].pop_front();
				}
			}
		}
	}

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

	void copy(double * buffer, list<double> & vals)
	{
		for (int i = 0; i < MAX_BUFFER; i++)
		{
			buffer[i] = 0;
		}


		list<double>::iterator iter;
		int i = MAX_BUFFER;

		for (iter = vals.end(); iter != vals.begin(); --iter)
		{
			buffer[--i] = *iter;
		}
	}

	void renderOutToFile()
	{
		printf("Writing Trace frame\n");
		mglGraph graph;

		graph.SetOrigin(0,0,0);

		graph.SetRange('y', _minY, _maxY);


		map<string, list<double> >::iterator iter;

		int plotCount = 0;
		for (iter = _traces.begin(); iter != _traces.end(); ++iter)
		{
			string name = (*iter).first;
			double buffer[MAX_BUFFER];
			copy(buffer, (*iter).second);
			mglData data(MAX_BUFFER, buffer);

			graph.SubPlot(2,2,plotCount++,name.c_str());
			graph.Box();
			graph.Title(name.c_str());
			graph.Plot(data);

		}

		char buf[64];
		sprintf(buf, "%05d", _fileCount);
		string filename = _fullOutputPath + "/trace_" + buf;
		graph.WritePNG((filename + ".png").c_str(), "");
		graph.WriteEPS((filename + ".eps").c_str(), "");

		_fileCount++;
		printf("Writing Trace frame - Done\n");
	}

public:
	TraceRenderingDataStream(string outputPath, int width,
							int height, int dimensionToPlot,
							int outputInterval, int outputSteps,
							double deltaT,
							double minY,
							double maxY):
		_outputPath(outputPath),
		_width(width),
		_height(height),
		_dimensionToPlot(dimensionToPlot),
		_outputInterval(outputInterval),
		_outputSteps(outputSteps),
		_deltaT(deltaT),
		_currentStep(0),
		_fileCount(1),
		_minY(minY),
		_maxY(maxY)
	{
		createDirs();
		allocateBuffer();
	}

	virtual void write(StateSpace * data, int width, int height)
	{
		buffer(data);
		_currentStep++;

		if (_currentStep % _outputSteps == 0)
		{
			renderOutToFile();
		}
	}

	virtual void waitToDrain() { }

	virtual ~TraceRenderingDataStream() {};

};


#endif /* TRACERENDERINGDATASTREAM_CUH_ */
