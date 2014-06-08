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

CostMap SumSqCost::SCompute(const Params &pred, const Params &labels)
{
	if (_cuImpl)
		return _cuImpl->Compute(pred, labels);

	Real cost = (labels.GetHostMatrix() - pred.GetHostMatrix()).squaredNorm();

	return CostMap{ { CostMap::PRIMARY_NAME, cost } };
}

Params SumSqCost::SComputeGrad(const Params &pred, const Params &labels)
{
	if (_cuImpl)
		return _cuImpl->ComputeGrad(pred, labels);

	return Params(pred, (pred.GetHostMatrix() - labels.GetHostMatrix())  / pred.Cols);
}

void SumSqCost::OnInitCudaDevice(int deviceId)
{
	_cuImpl.reset(new CuSumSqCost(deviceId));
}

void BindStruct(const aser::CStructBinder &binder, SumSqCost &cost)
{
    BindStruct(binder, (SimpleCost&)cost);
}

AXON_SERIALIZE_DERIVED_TYPE(ICost, SumSqCost, SumSqCost);


