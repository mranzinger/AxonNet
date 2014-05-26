/*
 * cumath_functions.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#include "cumath_unary_functions.cuh"
#include "cumath_binary_functions.cuh"

template<typename UnaryFn>
__global__ void ApplyUnaryFn(const Real *pVecSrc, Real *pVecTarget,
						     unsigned int rows, unsigned int cols, UnaryFn fn)
{
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= rows || x >= cols)
		return;

	pVecTarget[y * cols + x] = fn(pVecSrc[y * cols + x]);
}




