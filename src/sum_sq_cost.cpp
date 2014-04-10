#include "sum_sq_cost.h"

using namespace std;

Real SumSqCost::Compute(const Params &pred, const Params &labels)
{
	return (labels.Data - pred.Data).squaredNorm() / pred.Data.cols();
}

Params SumSqCost::ComputeGrad(const Params &pred, const Params &labels)
{
	return Params(pred, (pred.Data - labels.Data)  / pred.Data.cols());
}

void BindStruct(const axon::serialization::CStructBinder &, SumSqCost&) { }

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);
