/*
 * cumath_unary_functions.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#define __CUDACC__
#include <cuda_runtime_api.h>

#include "cudev_helper.cuh"
#include "math_defines.h"

// Basic Math Functions
struct CuSquare
{
	__device__ Real operator()(Real val) const
	{
		return val * val;
	}
};
struct CuSqrt
{
	__device__ Real operator()(Real val) const
	{
		return sqrt(val);
	}
};
struct CuLn
{
	__device__ Real operator()(Real val) const
	{
		return log(val);
	}
};
struct CuExp
{
	__device__ Real operator()(Real val) const
	{
		return exp(val);
	}
};
template<int Pow>
struct CuIntPow
{
	__device__ Real operator()(Real val) const
	{
		if (Pow == 0)
			return 1.0f;

		Real comp = val;
		for (int i = 1; i < Pow; ++i)
			comp = comp * val;
		return comp;
	}
};
struct CuUnaryScale
{
	Real Scale;

	CuUnaryScale(Real scale) : Scale(scale) { }

	__device__ Real operator()(Real val) const
	{
		return Scale * val;
	}
};

// Neural Network Functions
struct CuLogistic
{
	__device__ Real operator()(Real val) const
	{
		return 1.0f / (1.0f + exp(-val));
	}
};
struct CuLogisticDerivativeCalc
{
	__device__ Real operator()(Real val) const
	{
		return val * (1.0f - val);
	}
};
struct CuLogisticDerivativeRaw
{
	CuLogistic _logFn;
	CuLogisticDerivativeCalc _logDer;

	__device__ Real operator()(Real val) const
	{
		const Real comp = _logFn(val);

		return _logDer(comp);
	}
};
struct CuRectifier
{
	__device__ Real operator()(Real val) const
	{
		return val > 0.0f ? val : 0.0f;
	}
};
struct CuRectifierDerivative
{
	__device__ Real operator()(Real val) const
	{
		return val > 0.0f ? val : 0.0f;
	}
};
struct CuHardTanh
{
	__device__ Real operator()(Real val) const
	{
		if (val < -1)
			return -1;
		if (val > 1)
			return 1;
		return val;
	}
};
struct CuHardTanhDerivative
{
	__device__ Real operator()(Real val) const
	{
		if (val <= -1.f || val >= 1.f)
			return 0;
		return 1;
	}
};
struct CuSoftplus
{
	__device__ Real operator()(Real val) const
	{
		return log(1 + exp(val));
	}
};
struct CuSoftplusDerivativeRaw
{
	CuLogistic _logistic;

	__device__ Real operator()(Real val) const
	{
		return _logistic(val);
	}
};
struct CuTanh
{
	__device__ Real operator()(Real val) const
	{
		const Real e = exp(-val);

		return (1.0f - e) / (1.0f + e);
	}
};
struct CuTanhDerivativeCalc
{
	CuSquare _square;

	__device__ Real operator()(Real val) const
	{
		return 1.0f - _square(val);
	}
};
struct CuTanhDerivativeRaw
{
	CuTanh _tanh;
	CuTanhDerivativeCalc _calc;

	__device__ Real operator()(Real val) const
	{
		return _calc(_tanh(val));
	}
};
struct CuConstant
{
    Real _val;

    CuConstant(Real val) : _val(val) { }

    __device__ Real operator()(Real val) const
    {
        return _val;
    }
};
struct CuIdentity
{
    __device__ Real operator()(Real val) const
    {
        return val;
    }
};
struct CuZero
{
    __device__ Real operator()(Real val) const
    {
        return 0.0f;
    }
};
