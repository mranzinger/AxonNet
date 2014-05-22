#include "sum_sq_cost.h"

using namespace std;

SumSqCost::SumSqCost()
    : SumSqCost("", "")
{
}

SumSqCost::SumSqCost(std::string inputName)
    : SumSqCost(move(inputName), "")
{
}

SumSqCost::SumSqCost(std::string inputName, std::string labelName)
    : SimpleCost(move(inputName), move(labelName))
{
}

Real SumSqCost::SCompute(const Params &pred, const Params &labels)
{
	return (labels.Data - pred.Data).squaredNorm() / pred.Data.cols();
}

Params SumSqCost::SComputeGrad(const Params &pred, const Params &labels)
{
	return Params(pred, (pred.Data - labels.Data)  / pred.Data.cols());
}

void BindStruct(const aser::CStructBinder &binder, SumSqCost &cost)
{
    BindStruct(binder, (SimpleCost&)cost);
}

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);
