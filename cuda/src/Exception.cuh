/*
 * Exception.cuh
 *
 *  Created on: 03/05/2013
 *      Author: matt
 */

#ifndef EXCEPTION_CUH_
#define EXCEPTION_CUH_

class Exception
{
private:
	const char * _msg;
public:
	Exception(const char * msg) : _msg(msg){};
};


#endif /* EXCEPTION_CUH_ */
