/*
 * cudev_helper.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#ifndef _CUDA_COMPILE_
struct cuVec_t
{
	unsigned int x, y, z;
} blockIdx, blockDim, threadIdx;
#endif


