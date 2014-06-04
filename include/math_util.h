#pragma once

#include "math_defines.h"
#include "dll_include.h"

#ifndef _CUDA_COMPILE_
#include "thread/thread_pool.h"
#else
class CThreadPool;
#endif

NEURAL_NET_API void InitializeWeights(Vector &vec, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(RMatrix &mat, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(CMatrix &mat, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(Real *iter, Real *end, Real mean, Real stdDev);

NEURAL_NET_API void FanInitializeWeights(Vector &vec);
NEURAL_NET_API void FanInitializeWeights(RMatrix &mat);
NEURAL_NET_API void FanInitializeWeights(CMatrix &mat);
NEURAL_NET_API void FanInitializeWeights(Real *iter, Real *end, int wtSize = -1);

NEURAL_NET_API CThreadPool &GetThreadPool();

template<typename T>
T Square(T val)
{
	return val * val;
}

inline Vector SquareV(const Vector &v)
{
	return v.unaryExpr(&Square<Real>);
}

#ifndef _CUDA_COMPILE_
template<typename IterType>
class Range
{
private:
	IterType _begin, _end;

public:
	Range(IterType begin, IterType end)
		: _begin(std::move(begin)), _end(std::move(end)) { }

	IterType begin() const {
		return _begin;
	}
	IterType end() const {
		return _end;
	}
};

template<typename IterType>
Range<IterType> make_range(IterType begin, IterType end)
{
	return Range<IterType>(std::move(begin), std::move(end));
}
#endif

inline void MaxBinarize(CMatrix &mat)
{
	// Since the matrix is column major, it is more efficient to
	// traverse by row and then by column
	for (int col = 0, cEnd = mat.cols(); col < cEnd; ++col)
	{
		Real m = mat(0, col);
		size_t bestIdx = 0;

		for (int row = 1, rEnd = mat.rows(); row < rEnd; ++row)
		{
			Real v = mat(row, col);

			if (v > m)
			{
				m = v;
				bestIdx = row;
			}
		}

		for (int row = 0, rEnd = mat.rows(); row < rEnd; ++row)
		{
			if (row == bestIdx)
				mat(row, col) = 1;
			else
				mat(row, col) = 0;
		}
	}
}

inline size_t EqCount(const CMatrix &a, const CMatrix &b)
{
	size_t ret = 0;

	for (int col = 0, cEnd = a.cols(); col < cEnd; ++col)
	{
		if (a.col(col) == b.col(col))
			++ret;
	}

	return ret;
}

