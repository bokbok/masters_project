/*
 * UITraceDataStream.cuh
 *
 *  Created on: 09/06/2013
 *      Author: matt
 */

#ifndef UITRACEDATASTREAM_CUH_
#define UITRACEDATASTREAM_CUH_

#include "../DataStream.cuh"
#include "../../visual/Trace.cuh"
#include <mgl2/mgl.h>
//#include <mgl2/glut.h>
//#include <mgl2/qt.h>
//#include <pthread.h>


class UITraceDataStream : public DataStream
{
private:
	Trace _trace;
	int _currentStep, _renderSteps;
	mglGraph _graph;

	//pthread_t _thread;


public:
	UITraceDataStream(int width,
			int height,
			int dimensionToPlot,
			int renderSteps,
			int outputInterval,
			double deltaT,
			double minY,
			double maxY) :
		_trace(width, height, dimensionToPlot, outputInterval, deltaT, minY, maxY, 512),
		_currentStep(0),
		_renderSteps(renderSteps),
		_graph()
	{
	}

	virtual void write(Buffer * data)
	{
		data->checkOut();
		_trace.buffer(data->data());
		_currentStep++;

		if (_currentStep % _renderSteps == 0)
		{
			_trace.render(_graph);
			_graph.ShowImage("View");
		}
		data->release();
	}

	virtual void waitToDrain()
	{

	}


	virtual ~UITraceDataStream() {};
};


#endif /* UITRACEDATASTREAM_CUH_ */
