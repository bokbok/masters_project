/*
 * DataStream.hpp
 *
 *  Created on: 09/05/2013
 *      Author: matt
 */

#ifndef DATASTREAM_HPP_
#define DATASTREAM_HPP_
#include "../Buffer.cuh"

class DataStream
{
public:
	virtual void write(Buffer * data) = 0;
	virtual void waitToDrain() = 0;

	virtual ~DataStream() {};
};


#endif /* DATASTREAM_HPP_ */
