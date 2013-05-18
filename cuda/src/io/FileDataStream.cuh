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
#include <vector>

using namespace std;

class FileDataStream : public DataStream
{
private:
	string _path;
	vector<int> _dimensions;

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
	FileDataStream(string path, vector<int> dimensions):
		_path(path), _dimensions(dimensions)
	{
		open();
	}

	virtual void waitToDrain()
	{
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

				for (int col = 0; col < _dimensions.size(); col++)
				{
					double val = state[_dimensions[col]];
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
