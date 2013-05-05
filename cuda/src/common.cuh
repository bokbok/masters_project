/*
 * common.cuh
 *
 *  Created on: 03/05/2013
 *      Author: matt
 */

#ifndef COMMON_CUH_
#define COMMON_CUH_
#include <stdio.h>

//const int MAX_EQUATIONS=40;

#define CHECK_BOUNDS(var, max) \
		if (var > max || var < 0) \
		{ \
			printf("Array index out of bounds!"); \
		}


#endif /* COMMON_CUH_ */
