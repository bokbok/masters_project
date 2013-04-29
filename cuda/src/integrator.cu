#include "integrator.cuh"

Integrator::Integrator(int meshPointsX, int meshPointsY, double deltaT, int flushSteps):
	_meshPointsX(meshPointsX),
	_meshPointsY(meshPointsY),
	_deltaT(deltaT),
	_flushSteps(flushSteps)
{
	allocateMesh();
};

void Integrator::stepAndFlush(ostream & out, int steps)
{
	// iterator for x - 1
	// iterator for x - 2
	// iterator for x + 1
	// iterator for x + 2

	// iterator for y - 1
	// iterator for y - 2
	// iterator for y + 1
	// iterator for y + 2

	// in tuple zip iterator

}

void Integrator::allocateMesh()
{
	int size = NUM_EQUATIONS * _meshPointsX * _meshPointsY * _flushSteps;
	deviceMesh.resize(size, 0);
}
