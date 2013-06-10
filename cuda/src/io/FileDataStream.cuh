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
#include <map>

using namespace std;

class FileDataStream : public DataStream
{
private:
	string _path;
	map<string, int> _dimensions;

	ofstream _out;

	void open()
	{
		_out.open(_path.c_str());
		_out.precision(15);
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

		_out << endl;
	}

public:
	FileDataStream(string path, map<string, int> dimensions):
		_path(path), _dimensions(dimensions)
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

		_out << "t=" << (*data)[0].t();

		for (int x = 0; x < data->width(); x++)
		{
			for (int y = 0; y < data->height(); y++)
			{
				StateSpace & state = (*data)[x + y * data->width()];
				_out << "\n" << "(" << x << "," << y << "):";

				for (iter = _dimensions.begin(); iter != _dimensions.end(); ++iter)
				{
					double val = state[iter->second];
					_out << val << " ";
				}
			}
		}

		_out << '\n';
		data->release();
	}

	virtual ~FileDataStream()
	{
		close();
	};

};



#endif /* FILEDATASTREAM_CUH_ */
