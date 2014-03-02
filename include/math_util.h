#pragma once

#include <Eigen\Dense>

#include "dll_include.h"

typedef float Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Map<Matrix> Map;

NEURAL_NET_API void InitializeWeights(Vector &vec, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(Matrix &mat, Real mean, Real stdDev);
NEURAL_NET_API void InitializeWeights(Real *iter, Real *end, Real mean, Real stdDev);

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