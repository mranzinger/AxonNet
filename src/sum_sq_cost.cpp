#include "sum_sq_cost.h"

using namespace std;

Real SumSqCost::Compute(const Vector &pred, const Vector &labels)
{
	return (labels - pred).sum();
}

Vector SumSqCost::ComputeGrad(const Vector &pred, const Vector &labels)
{
	Vector errs = labels - pred;

	return move(errs);
}

void BindStruct(const axon::serialization::CStructBinder &, SumSqCost&) { }

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);