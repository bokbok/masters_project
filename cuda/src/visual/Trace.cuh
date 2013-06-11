/*
 * Trace.cuh
 *
 *  Created on: 09/06/2013
 *      Author: matt
 */

#ifndef TRACE_CUH_
#define TRACE_CUH_

#include <mgl2/mgl.h>
#include <list>
#include <map>

using namespace std;

class Trace
{
private:
	int _width, _height;
	int _dimensionToPlot;
	int _outputInterval;
	int _bufferSize;
	double _deltaT;

	double _minY,_maxY;
	map< string, list<double> > _traces;

	static const int MAX_BUFFER = 2048;

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

	void copy(double * buffer, list<double> & vals)
	{
		for (int i = 0; i < _bufferSize; i++)
		{
			buffer[i] = 0;
		}


		list<double>::iterator iter;
		int i = _bufferSize;

		for (iter = vals.end(); iter != vals.begin(); --iter)
		{
			buffer[--i] = *iter;
		}
	}


public:
	Trace(int width, int height, int dimension, int outputInterval, double deltaT, double minY, double maxY, int bufferSize = MAX_BUFFER) :
		_width(width),
		_height(height),
		_dimensionToPlot(dimension),
		_outputInterval(outputInterval),
		_deltaT(deltaT),
		_minY(minY),
		_maxY(maxY),
		_bufferSize(bufferSize)
	{
		allocateBuffer();
	}

	void buffer(StateSpace * data)
	{
		for (int x = 0; x < _width; x+= _outputInterval)
		{
			for (int y = 0; y < _height; y+= _outputInterval)
			{
				string key = traceKey(x, y);
				_traces[key].push_back(data[x + _width * y][_dimensionToPlot]);
				if (_traces[key].size() > _bufferSize)
				{
					_traces[key].pop_front();
				}
			}
		}
	}

	void push(StateSpace * data)
	{
		buffer(data);
	}

	void render(mglGraph & graph)
	{
		graph.SetOrigin(0,0,0);

		graph.SetRange('y', _minY, _maxY);

		map<string, list<double> >::iterator iter;

		int plotCount = 0;
		for (iter = _traces.begin(); iter != _traces.end(); ++iter)
		{
			string name = (*iter).first;
			double buffer[_bufferSize];
			copy(buffer, (*iter).second);
			mglData data(_bufferSize, buffer);

			graph.SubPlot(_width / _outputInterval, _width / _outputInterval, plotCount++,name.c_str());
			graph.Box();
			graph.Title(name.c_str());
			graph.Plot(data);

		}

	}

	void renderFile(string filenameWithoutExtension)
	{
		printf("Writing Trace frame\n");
		mglGraph graph;

		render(graph);

		graph.WritePNG((filenameWithoutExtension + ".png").c_str(), "");
		//graph.WriteEPS((filenameWithoutExtension + ".eps").c_str(), "");

		printf("Writing Trace frame - Done\n");
	}


};


#endif /* TRACE_CUH_ */
