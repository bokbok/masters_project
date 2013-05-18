/*
 * FileDataStream.cuh
 *
 *  Created on: 14/05/2013
 *      Author: matt
 */

#ifndef FILEDATASTREAM_CUH_
#define FILEDATASTREAM_CUH_

#include <string>
#include <fstream>

using namespace std;

class FileDataStream : public DataStream
{
private:
	string _path;

	ofstream _out;



	void open()
	{
		_out.open(_path.c_str());
	}

	void close()
	{
		_out.close();
	}

public:
	FileDataStream(string path):
		_path(path)
	{
		open();
	}

	virtual void write(StateSpace * data, int width, int height)
	{
		_out << "t=" << data[0].t();
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				StateSpace & state = data[x + y * width];
				_out << endl << "(" << x << "," << y << "):";
				for (int dim = 0; dim < state.numDimensions(); dim++)
				{
					double val = state[dim];
					_out << val << " ";
				}
			}
		}
		_out << endl;

	}

	virtual ~FileDataStream()
	{
		close();
	};

};



#endif /* FILEDATASTREAM_CUH_ */
