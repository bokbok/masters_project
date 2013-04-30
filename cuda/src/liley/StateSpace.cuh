/*
 * StateSpace.cuh
 *
 *  Created on: 29/04/2013
 *      Author: matt
 */

#ifndef STATESPACE_CUH_
#define STATESPACE_CUH_

class StateSpace
{
public:
	StateSpace()
	{

	}

	__device__
	double * stateVars()
	{
		return _stateVars;
	}


	__device__
	int dims()
	{
		return _dims;
	}

	static StateSpace zero;

private:
	double * _stateVars;
	int _dims;
};


#endif /* STATESPACE_CUH_ */
