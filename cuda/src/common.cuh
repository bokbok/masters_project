/*
 * common.cuh
 *
 *  Created on: 03/05/2013
 *      Author: matt
 */

#ifndef COMMON_CUH_
#define COMMON_CUH_
#include <stdio.h>
#include <vector>
#include <map>


#define BOUNDS_CHECK

#ifndef _BOUNDS_CHECK
	#define CHECK_BOUNDS(var, max)
#else
	#define CHECK_BOUNDS(var, max) \
			if (var > max || var < 0) \
			{ \
				printf("Array index out of bounds %i!", var); \
			}
#endif

const int BLOCK_SIZE = 10;

#endif /* COMMON_CUH_ */
