/*
 * ParameterMesh.cuh
 *
 *  Created on: 23/07/2013
 *      Author: matt
 */

#ifndef PARAMETERMESH_CUH_
#define PARAMETERMESH_CUH_

#include "../ParameterSpace.cuh"

template <class M>
class ParameterMesh
{
public:
	ParameterMesh()
	{

	}

	virtual ParameterSpace paramsAt(int x, int y) = 0;

	virtual ~ParameterMesh()
	{

	}
};


#endif /* PARAMETERMESH_CUH_ */
