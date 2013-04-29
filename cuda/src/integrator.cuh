/*
 * integrator.cuh
 *
 *  Created on: 28/04/2013
 *      Author: matt
 */

#ifndef INTEGRATOR_CUH_
#define INTEGRATOR_CUH_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <ostream>

using namespace std;

const int NUM_EQUATIONS = 14;

class Integrator
{
public:
	Integrator(int meshPointsX, int meshPointsY, double deltaT, int flushSteps);
	void stepAndFlush(ostream & out, int steps);

private:
	int _meshPointsX, _meshPointsY;
	int _flushSteps;
	double _deltaT;
	thrust::device_vector<double> deviceMesh;

	void allocateMesh();



};




#endif /* INTEGRATOR_CUH_ */
