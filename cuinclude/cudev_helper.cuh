/*
 * cudev_helper.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime_api.h>

#ifndef _CUDA_COMPILE_
struct cuVec_t
{
	unsigned int x, y, z;
} blockIdx, blockDim, threadIdx;
#endif



