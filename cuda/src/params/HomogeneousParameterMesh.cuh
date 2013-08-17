/*
 * HomogeneousParameterMesh.cuh
 *
 *  Created on: 23/07/2013
 *      Author: matt
 */

#ifndef HOMOGENEOUSPARAMETERMESH_CUH_
#define HOMOGENEOUSPARAMETERMESH_CUH_

#include "ParameterMesh.cuh"
#include <vector>

using namespace std;

template <class M>
class HomogeneousParameterMesh : public ParameterMesh<M>
{
private:
	ParameterSpace _prototype;

public:
	HomogeneousParameterMesh(ParameterSpace & prototype) : _prototype(prototype)
	{
		M::precalculate(_prototype);
	}

	virtual ParameterSpace paramsAt(int x, int y)
	{
		return _prototype;
	}

	virtual ~HomogeneousParameterMesh()
	{
	}

	virtual string describe()
	{
		return "Homogeneous parameters";
	}
};



#endif /* HOMOGENEOUSPARAMETERMESH_CUH_ */
