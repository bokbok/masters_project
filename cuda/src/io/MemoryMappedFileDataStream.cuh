/*
 * MemoryMappedFileDataStream.cuh
 *
 *  Created on: 25/05/2013
 *      Author: matt
 */

#ifndef MEMORYMAPPEDFILEDATASTREAM_CUH_
#define MEMORYMAPPEDFILEDATASTREAM_CUH_

#include <string>
#include <map>

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

using namespace std;

const int CHUNK_SIZE = 1024 * 1024 * 1024 * sizeof(char); // 100 meg

class MemoryMappedFileDataStream : public DataStream
{
private:
	string _path;
	map<string, int> _dimensions;

	int _handle;
	char * _contents;

	size_t _currentLength;
	size_t _writeLength;
	char * _currentPos;

	void open()
	{
		printf("Opening file to write\n");
		_currentLength = CHUNK_SIZE;
		_writeLength = 0;
		_handle = ::open(_path.c_str(), O_RDWR | O_TRUNC | O_CREAT, 0666);
		lseek(_handle, _currentLength - 1, SEEK_SET);
		::write(_handle, "", 1);
		_contents = (char *) mmap(0, _currentLength, PROT_READ | PROT_WRITE, MAP_SHARED, _handle, 0);
		if (_contents == MAP_FAILED)
		{
			printf("Failed to open file for writing\n");
		}
		_currentPos = _contents;
		printf("Opened file to write\n");
	}

	void close()
	{
		munmap(_contents, _currentLength);
		ftruncate(_handle, _writeLength);
		::close(_handle);
	}

	void write(string data)
	{
		if (_writeLength + data.length() > _currentLength)
		{
			grow();
		}
		memcpy(_contents + _writeLength, data.c_str(), data.length());

		_writeLength += data.length();
	}

	void grow()
	{
		long newLength = _currentLength + CHUNK_SIZE;

		lseek(_handle, newLength - 1, SEEK_SET);
		::write(_handle, "", 1);

		_contents = (char *) mremap(_contents, _currentLength, newLength, MREMAP_MAYMOVE);
		_currentLength = newLength;
	}

	void writeHeaders()
	{
		map<string, int>::iterator iter;

		for(iter = _dimensions.begin(); iter != _dimensions.end(); ++iter)
		{
			write(iter->first + " ");
		}
		write(string("\n"));
	}

	string convert(int i)
	{
		char buf[32];

		sprintf(buf, "%i", i);

		return buf;
	}

	string convert(double f)
	{
		char buf[32];

		sprintf(buf, "%f", f);

		return buf;
	}

public:
	MemoryMappedFileDataStream(string path, map<string, int> dimensions):
		_path(path), _dimensions(dimensions)
	{
		open();
		writeHeaders();
	}

	virtual void waitToDrain()
	{
	}

	virtual void write(StateSpace * data, int width, int height)
	{
		map<string, int>::iterator iter;

		write(string("t=") + convert(data[0].t()));

		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				StateSpace & state = data[x + y * width];
				write("\n(" + convert(x) + "," + convert(y) + "):");

				for (iter = _dimensions.begin(); iter != _dimensions.end(); ++iter)
				{
					double val = state[iter->second];
					write(convert(val) + " ");
				}
			}
		}

		write("\n");
	}

	virtual ~MemoryMappedFileDataStream()
	{
		close();
	};

};

#endif /* MEMORYMAPPEDFILEDATASTREAM_CUH_ */
