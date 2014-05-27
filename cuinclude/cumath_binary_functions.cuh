/*
 * cumath_binary_functions.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#define __CUDACC__
#include <cuda_runtime_api.h>

#include "cudev_helper.cuh"
#include "math_defines.h"

struct CuPlus
{
	__device__ Real operator()(Real a, Real b) const
	{
		return a + b;
	}
};
struct CuMinus
{
	__device__ Real operator()(Real a, Real b) const
	{
		return a - b;
	}
};
struct CuMultiply
{
	__device__ Real operator()(Real a, Real b) const
	{
		return a * b;
	}
};
struct CuDivide
{
	__device__ Real operator()(Real a, Real b) const
	{
		return a / b;
	}
};
struct CuPow
{
	__device__ Real operator()(Real a, Real b) const
	{
		return pow(a, b);
	}
};
struct CuAddScaledBinary
{
	Real ScaleA, ScaleB;

	CuAddScaledBinary(Real scaleA, Real scaleB)
		: ScaleA(scaleA), ScaleB(scaleB)
	{
	}

	__device__ Real operator()(Real a, Real b) const
	{
		return ScaleA * a + ScaleB * b;
	}
};
struct CuMulScaledBinary
{
	Real Scale;

	CuMulScaledBinary(Real scale)
		: Scale(scale)
	{
	}

	__device__ Real operator()(Real a, Real b) const
	{
		return Scale * a * b;
	}
};

