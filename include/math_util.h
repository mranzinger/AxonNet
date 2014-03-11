#pragma once

#include <Eigen/Dense>

#include "dll_include.h"

typedef float Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Map<Matrix> Map;

NEURAL_NET_API void InitializeWeights(Vector &vec, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(Matrix &mat, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(Real *iter, Real *end, Real mean, Real stdDev);

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

inline void MaxBinarize(Vector &v)
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
