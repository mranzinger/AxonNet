/*
 * cumath_binary_functions.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#include "cumath_traits.cuh"
#include <float.h>

struct CuPlus
{
    __device__ Real NullValue() const
    {
        return 0.0f;
    }

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
    __device__ Real NullValue() const
    {
        return 1.0f;
    }

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
struct CuMax
{
    __device__ Real NullValue() const
    {
        return -FLT_MAX;
    }
	__device__ Real operator()(Real a, Real b) const
	{
		return max(a, b);
	}
};
struct CuMin
{
    __device__ Real NullValue() const
    {
        return FLT_MAX;
    }
	__device__ Real operator()(Real a, Real b) const
	{
		return min(a, b);
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
struct CuTakeLeft
{
    __device__ Real operator()(Real a, Real b) const
    {
        return a;
    }
};
struct CuTakeRight
{
    __device__ Real operator()(Real a, Real b) const
    {
        return b;
    }
};
struct CuSquaredDiff
{
    __device__ Real operator()(Real a, Real b) const
    {
        const Real diff = a - b;

        return diff * diff;
    }
};
struct CuScaledDiff
{
    Real _scale;
    CuScaledDiff(Real scale) : _scale(scale) { }

    __device__ Real operator()(Real a, Real b) const
    {
        return _scale * (a - b);
    }
};
