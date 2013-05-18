/*
 * DataStream.hpp
 *
 *  Created on: 09/05/2013
 *      Author: matt
 */

#ifndef DATASTREAM_HPP_
#define DATASTREAM_HPP_
#include "../StateSpace.cuh"

class DataStream
{
public:
	virtual void write(StateSpace * data, int width, int height) = 0;
	virtual void waitToDrain() = 0;

	virtual ~DataStream() {};
};


#endif /* DATASTREAM_HPP_ */
