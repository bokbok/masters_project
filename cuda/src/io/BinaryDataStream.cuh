/*
 * BinaryDataStream.cuh
 *
 *  Created on: 12/08/2013
 *      Author: matt
 */

#ifndef BINARYDATASTREAM_CUH_
#define BINARYDATASTREAM_CUH_

#include <string>
#include <fstream>
#include <map>

using namespace std;

class BinaryDataStream : public DataStream
{
private:
	string _path;
	map<string, int> _dimensions;
	int _meshSize;

	ofstream _out;

	void open()
	{
		_out.open(_path.c_str(), ios::out | ios::binary);
		if (!_out.is_open())
		{
			cerr << "File failed to open: " << _path << endl;
		}
	}

	void close()
	{
		_out.close();
	}

	void writeHeaders()
	{
		map<string, int>::iterator iter;

		for(iter = _dimensions.begin(); iter != _dimensions.end(); ++iter)
		{
			_out << iter->first << " ";
		}

		_out << _meshSize;

		_out << endl;
	}

public:
	BinaryDataStream(string path, map<string, int> dimensions, int meshSize):
		_path(path), _dimensions(dimensions), _meshSize(meshSize)
	{
		open();
		writeHeaders();
	}

	virtual void waitToDrain()
	{
	}

	virtual void write(Buffer * data)
	{
		data->checkOut();
		map<string, int>::iterator iter;

		double t = (*data)[0].t();
		_out.write((char *)&t, sizeof(double));

		for (int x = 0; x < data->width(); x++)
		{
			for (int y = 0; y < data->height(); y++)
			{
				StateSpace & state = (*data)[x + y * data->width()];
				for (iter = _dimensions.begin(); iter != _dimensions.end(); ++iter)
				{
					double val = state[iter->second];
					_out.write((char *)&val, sizeof(double));

				}
			}
		}
		data->release();
	}

	virtual ~BinaryDataStream()
	{
		close();
	};

};


#endif /* BINARYDATASTREAM_CUH_ */
