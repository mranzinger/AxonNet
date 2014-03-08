#include "sum_sq_cost.h"

using namespace std;

Real SumSqCost::Compute(const Vector &pred, const Vector &labels)
{
	return (labels - pred).squaredNorm();
}

Vector SumSqCost::ComputeGrad(const Vector &pred, const Vector &labels)
{
	Vector errs = pred - labels;

	return move(errs);
}

void BindStruct(const axon::serialization::CStructBinder &, SumSqCost&) { }

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);