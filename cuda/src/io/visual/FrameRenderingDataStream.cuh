/*
 * FrameRenderingDataStream.cuh
 *
 *  Created on: 29/05/2013
 *      Author: matt
 */

#ifndef FRAMERENDERINGDATASTREAM_CUH_
#define FRAMERENDERINGDATASTREAM_CUH_

#include "../DataStream.cuh"
#include "../../StateSpace.cuh"

#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <mgl2/config.h>

#define MGL_HAVE_RVAL   0

#include <mgl2/mgl.h>

#include <vector>

using namespace std;

class FrameRenderingDataStream : public DataStream
{
private:
	string _outputPath, _fullOutputPath;
	int _dimensionToPlot;

	int _outputInterval;

	double * _mesh;

	int _currentStep;
	int _meshSize, _width, _height;

	int _fileCount;

	double _latestT;
	double _restingVal;

	void prepareOutputPath()
	{
		mkdir(_outputPath.c_str(), 0777);
		_fullOutputPath = _outputPath + "/rms";
		mkdir(_fullOutputPath.c_str(), 0777);
	}

	void allocateMesh(int width, int height)
	{
		_meshSize = width * height;

		for (int i = 0; i < _outputInterval; i++)
		{
			_mesh = new double[_meshSize];

			zeroMesh(_mesh);
		}
	}

	void deallocateMesh()
	{
		delete[] _mesh;
	}

	void zeroMesh(double * mesh)
	{
		for (int i = 0; i < _meshSize; i++)
		{
			mesh[i] = 0.0;
		}
	}

	void writeRMSFrame()
	{
		printf("Writing RMS frame\n");
		double rms[_meshSize];
		for (int i = 0; i < _meshSize; i++)
		{
			rms[i] = (sqrt(_mesh[i] / (double)_outputInterval) - 7.5) / 15;
		}

		mglGraph graph;

		mglData data(_height, _width, rms);

		char buf[255];
		_fileCount++;
		sprintf(buf, "t = %f", _latestT);

		graph.Title(buf);
		graph.Rotate(50,30);
		graph.Light(true);
		graph.Box();
		graph.Surf(data);

		sprintf(buf, "%05d", _fileCount);
		string filename = _fullOutputPath + "/rms_" + buf + ".png";
		printf(filename.c_str());
		graph.WritePNG(filename.c_str(), "");
		printf("Writing RMS frame - Done\n");
	}

	void updateMesh(Buffer * data)
	{
		_latestT = (*data)[0].t();
		for (int i = 0; i < _meshSize; i++)
		{
			double val = (*data)[i][_dimensionToPlot] - _restingVal;
			_mesh[i] += val * val;
		}

		_currentStep++;
		_currentStep = _currentStep % _outputInterval;

		if (_currentStep == 0)
		{
			writeRMSFrame();
			zeroMesh(_mesh);
		}
	}

public:
	FrameRenderingDataStream(string outputPath, int width, int height, int dimensionToPlot, int outputInterval, double restingVal) :
		_outputPath(outputPath),
		_dimensionToPlot(dimensionToPlot),
		_outputInterval(outputInterval),
		_meshSize(-1),
		_currentStep(0),
		_width(width),
		_height(height),
		_fileCount(0),
		_latestT(0.0),
		_restingVal(restingVal)
	{
		prepareOutputPath();
		allocateMesh(width, height);
	}


	virtual void write(Buffer * data)
	{
		data->checkOut();
		updateMesh(data);
		data->release();
	}

	virtual void waitToDrain() {}

	virtual ~FrameRenderingDataStream()
	{
		deallocateMesh();
	};
};



#endif /* FRAMERENDERINGDATASTREAM_CUH_ */
