#include "sum_sq_cost.h"

using namespace std;

Real SumSqCost::Compute(const Vector &pred, const Vector &labels)
{
	return (labels - pred).sum();
}

Vector SumSqCost::ComputeError(const Vector &pred, const Vector &labels)
{
	Vector errs = labels - pred;

	return move(errs);
}

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);