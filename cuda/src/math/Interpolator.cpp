#include "Interpolator.hpp"
#include <stdio.h>
//#include <einspline/nubspline.h>

double * Interpolator::interpolateMesh()
{
	double * result = new double[_meshSize * _meshSize];
	double delta = ((double)_meshSize / (double)_samplePoints);

	printf(" %lf \n",delta);
	for (int x=0;x < _samplePoints; x++)
	{
		for (int y=0;y < _samplePoints; y++)
		{
			printf(" (%i, %i):%lf",x, y, _points[x + y * _samplePoints]);
		}
		printf("\n");
	}


	// populate in y direction first
	for (int ySamp = 0; ySamp < _samplePoints; ySamp++)
	{
		int y = ySamp * delta;
		for (int xSamp = 0; xSamp < _samplePoints; xSamp++)
		{
			int x = xSamp * delta;
			int xSampNext = xSamp + 1 > _samplePoints ? 0 : xSamp + 1;
			double segStart = _points[xSamp + ySamp * _samplePoints];
			double segEnd = _points[xSampNext + ySamp * _samplePoints];

			double deltaZ = segEnd - segStart;
			for (int seg = 0; seg < delta; seg++)
			{
				result[x + seg + ySamp * _meshSize] = segStart + deltaZ * (seg / delta);
			}
		}
	}

	// now fill in the x direction from what was computed in y
	for (int x = 0; x < _meshSize; x++)
	{
		for (int ySamp = 0; ySamp < _samplePoints; ySamp++)
		{
			int ySampNext = ySamp + 1 > _samplePoints ? 0 : ySamp + 1;
			double segStart = result[x + ySamp * _meshSize];
			double segEnd = result[x + ySampNext * _meshSize];

			double deltaZ = segEnd - segStart;

			for (int seg = 0; seg < delta; seg++)
			{
				result[x + (int)(ySamp * delta + seg) * _meshSize] = segStart + deltaZ * (seg / delta);
			}
		}
	}


	printf("*** Interpolated Values: \n");
//	for (int x = 0; x < _meshSize; x++)
//	{
//		for (int y = 0; y < _meshSize; y++)
//		{
//			printf(" (%i, %i):%lf",x, y, result[x + y * _meshSize]);
//		}
//		printf("\n");
//	}

	printf("\n");
	return result;
}

//double * Interpolator::interpolateMesh()
//{
//	printf("Here1\n");
//	double * result = new double[_meshSize * _meshSize];
//	double meshPoints[_samplePoints];
//	float data[_samplePoints * _samplePoints];
//
//	double delta = (_meshSize / _samplePoints);
//
//	int i = 0;
//	for (int i = 0; i < _samplePoints; i++)
//	{
//		for (int j = 0; j < _samplePoints; j++)
//		{
//			data[i + j * _samplePoints] = _points[i + j * _samplePoints];
//		}
//		meshPoints[i] = i * delta;
//		printf("%i:%f ", i, meshPoints[i]);
//	}
//
//
//	NUgrid* xGrid = create_general_grid(meshPoints, _samplePoints);
//	NUgrid* yGrid = create_general_grid(meshPoints, _samplePoints);
//	BCtype_s xBC, yBC;
//	xBC.lCode = FLAT;
//	yBC.lCode = FLAT;
//
//	NUBspline_2d_s * spline = create_NUBspline_2d_s(xGrid, yGrid, xBC, yBC, data);
//	if (!spline)
//	{
//		printf("Failed to generate spline!\n");
//	}
//	for (int y = 0; y < _meshSize; y++)
//	{
//		for (int x = 0; x < _meshSize; x++)
//		{
//			float val = 0;
//			eval_NUBspline_2d_s(spline, x, y, &val);
//			result[x + y * _meshSize] = val;
//
//			printf("%i, %i:%f \n", x, y, val);
//		}
//	}
//
//	destroy_Bspline(spline);
//
//	printf("Done");
//	return result;
//}
