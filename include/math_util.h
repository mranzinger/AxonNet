#pragma once

#include <Eigen/Dense>
#include <vector>

#include "dll_include.h"
#include "thread/thread_pool.h"

typedef float Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Real, 1, Eigen::Dynamic> RowVector;
typedef std::vector<Vector> MultiVector;
typedef Eigen::Map<Vector, Eigen::Aligned> MapVector;
typedef Eigen::Map<RowVector, Eigen::Aligned> MapRowVector;
typedef Eigen::Map<Vector, Eigen::Unaligned> UMapVector;
typedef Eigen::Map<RowVector, Eigen::Unaligned> UMapRowVector;
typedef Eigen::Map<Vector, 0, Eigen::OuterStride<>> StrideVec;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> CMatrix;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;

typedef Eigen::Map<CMatrix> CMap;
typedef Eigen::Map<RMatrix> RMap;

typedef Eigen::Map<CMatrix, Eigen::Unaligned> CUMap;
typedef Eigen::Map<RMatrix, Eigen::Unaligned> RUMap;

typedef Eigen::Map<CMatrix, 0, Eigen::OuterStride<>> CStrideMap;
typedef Eigen::Map<RMatrix, 0, Eigen::OuterStride<>> RStrideMap;

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
auto Square(const T &val) -> decltype(val * val)
{
	return val * val;
}

inline Vector SquareV(const Vector &v)
{
	return v.unaryExpr([](Real val) { return val * val; });
}

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

/*inline void MaxBinarize(Vector &v)
{
	Real m = v[0];
	size_t bestIdx = 0;

	for (size_t i = 1, end = v.size(); i < end; ++i)
	{
		if (v[i] > m)
		{
			m = v[i];
			bestIdx = i;
		}
	}

	for (size_t i = 0, end = v.size(); i < end; ++i)
	{
		if (i != bestIdx)
			v[i] = 0;
		else
			v[i] = 1;
	}
}*/

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


template<size_t History>
class WindowStats
{
	static_assert(History >= 2, "Cannot have a history with less than 2 elements");

private:
	Real _history[History];
	size_t _opIdx = 0;
	bool _full = false;

	Real _runningSum = 0;
	Real _runningSumSq = 0;

public:
	WindowStats()
	{
		memset(_history, 0, sizeof(_history));
	}

	void Append(Real val)
	{
		Real old = _history[_opIdx];

		_runningSum -= old;
		_runningSumSq -= Square(old);

		_history[_opIdx] = val;

		_runningSum += val;
		_runningSumSq += Square(val);

		++_opIdx;
		if (_opIdx == History)
		{
			_opIdx = 0;
			_full = true;
		}
	}

	bool Full() const
	{
		return _full;
	}

	Real Sum() const {
		return _runningSum;
	}

	Real Mean() const
	{
		if (_full)
			return _runningSum / History;
		else if (_opIdx > 0)
			return _runningSum / _opIdx;
		else
			return 0;
	}

	Real StdDev() const
	{
		size_t n;
		if (_full)
			n = History;
		else if (_opIdx > 0)
			n = _opIdx;
		else
			return 0;

		return sqrt((_runningSumSq / n) - Square(_runningSum / n));
	}

};

#include "matrix_math.h"

