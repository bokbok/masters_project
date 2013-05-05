#include <stdio.h>

using namespace std;

#include "StateSpace.cuh"

#include "Mesh.cuh"
#include "liley/Model.cuh"

int main(void)
{
	Mesh mesh(10, 10, 0.0001, 100);

	mesh.stepAndFlush(1, 0.0001, cout);
	mesh.stepAndFlush(1.0001, 0.0001, cout);

    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	return 0;
}
