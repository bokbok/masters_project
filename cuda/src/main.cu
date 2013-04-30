#include <boost/numeric/odeint.hpp>
#include <stdio.h>
#include "integrator.cuh"

using namespace std;

__global__ void runFn(double val)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

	printf("\nRunning for mesh point %d %d", x, y);
}

#include "Mesh.cuh"
#include "liley/StateSpace.cuh"

typedef Mesh<StateSpace> LileyMesh;

int main(void)
{
//	cout << "here" << endl;
//    dim3 threads(32, 6);
//    dim3 blocks(2, 2);
//
//    runFn<<<threads, blocks>>>(12.0);

	Integrator integrator(100, 100, 0.00001, 100);

	LileyMesh mesh(10, 10, 0.0001, 100);

	mesh.stepAndFlush(cout);
	mesh.stepAndFlush(cout);

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
