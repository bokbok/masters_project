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
#include <ctime>

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

	void prepareOutputPath()
	{
		mkdir(_outputPath.c_str(), 0777);
		double sysTime = time(0);

		char buf[600];
		sprintf(buf, "%i", (int) sysTime);
		_fullOutputPath = (_outputPath + "/" + buf);

		mkdir(_fullOutputPath.c_str(), 0777);
	}

	void allocateMesh(int width, int height)
	{
		_meshSize = width * height;

		for (int i = 0; i < _outputInterval; i++)
		{
			_mesh = new double[_meshSize];

			zeroMesh();
		}
	}

	void deallocateMesh()
	{
		delete[] _mesh;
	}

	void zeroMesh()
	{
		for (int i = 0; i < _meshSize; i++)
		{
			_mesh[i] = 0.0;
		}
	}

	void writeRMSFrame()
	{
		printf("Writing RMS frame\n");
		double rms[_meshSize];
		for (int i = 0; i < _meshSize; i++)
		{
			rms[i] = (100 - sqrt(_mesh[i] / (double)_outputInterval)) / 100;
		}

		mglGraph graph;

		mglData data(_height, _width, rms);

		char buf[255];
		_fileCount++;
		sprintf(buf, "t = %f", _latestT);

		graph.Title(buf);

		graph.Box();
		graph.Dens(data);

		sprintf(buf, "%i", _fileCount);
		string filename = _fullOutputPath + "/" + buf + ".png";
		printf(filename.c_str());
		graph.WritePNG(filename.c_str(), "");
		printf("Writing RMS frame - Done\n");
	}

	void updateMesh(StateSpace * data)
	{
		_latestT = data[0].t();
		for (int i = 0; i < _meshSize; i++)
		{
			_mesh[i] += data[i][_dimensionToPlot] * data[i][_dimensionToPlot];
		}

		_currentStep++;
		_currentStep = _currentStep % _outputInterval;

		if (_currentStep == 0)
		{
			writeRMSFrame();
			zeroMesh();
		}
	}

public:
	FrameRenderingDataStream(string outputPath, int width, int height, int dimensionToPlot, int outputInterval) :
		_outputPath(outputPath),
		_dimensionToPlot(dimensionToPlot),
		_outputInterval(outputInterval),
		_meshSize(-1),
		_currentStep(0),
		_width(width),
		_height(height),
		_fileCount(0),
		_latestT(0.0)
	{
		prepareOutputPath();
		allocateMesh(width, height);
	}


	virtual void write(StateSpace * data, int width, int height)
	{
		updateMesh(data);
	}

	virtual void waitToDrain() {}

	virtual ~FrameRenderingDataStream()
	{
		deallocateMesh();
	};
};



#endif /* FRAMERENDERINGDATASTREAM_CUH_ */
