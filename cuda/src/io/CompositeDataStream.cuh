/*
 * CompositeDataStream.cuh
 *
 *  Created on: 29/05/2013
 *      Author: matt
 */

#ifndef COMPOSITEDATASTREAM_CUH_
#define COMPOSITEDATASTREAM_CUH_

#include <vector>

using namespace std;

class CompositeDataStream : public DataStream
{
private:
	vector<DataStream *> _streams;

public:
	CompositeDataStream(vector<DataStream *> streams) :
		_streams(streams)
	{

	}

	virtual void write(Buffer * data)
	{
		vector<DataStream *>::iterator iter;

		data->checkOut();
		for (iter = _streams.begin(); iter != _streams.end(); ++iter)
		{
			(*iter)->write(data);
		}
		data->release();
	}

	virtual void waitToDrain()
	{
		vector<DataStream *>::iterator iter;

		for (iter = _streams.begin(); iter != _streams.end(); ++iter)
		{
			(*iter)->waitToDrain();
		}
	}

	virtual ~CompositeDataStream() {};

};


#endif /* COMPOSITEDATASTREAM_CUH_ */
