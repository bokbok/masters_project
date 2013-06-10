/*
 * ConvergenceMonitor.cuh
 *
 *  Created on: 03/06/2013
 *      Author: matt
 */

#ifndef CONVERGENCEMONITOR_CUH_
#define CONVERGENCEMONITOR_CUH_

#include "../DataStream.cuh"
#include <iostream>

using namespace std;

class ConvergenceMonitor : public DataStream
{
public:
	ConvergenceMonitor(){}

	virtual void write(Buffer * data)
	{
		int N = data->length();

		for (int i = 0; i < N; i++)
		{
			if ((*data)[i].nan())
			{
				cerr << "Convergence failed!" << endl;
				exit(-1);
			}
		}

	}

	virtual void waitToDrain() { }

};


#endif /* CONVERGENCEMONITOR_CUH_ */
