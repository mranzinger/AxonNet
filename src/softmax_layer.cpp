#include "softmax_layer.h"
#include "functions.h"
#include "logloss_cost.h"
#include "neural_net.h"

using namespace std;
using namespace axon::serialization;

Params SoftmaxLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	// Getting the max value and shifting the dimension is a neat trick to prevent overflow
	// NOTE: Undeflow may still occur though, but that is far less dangerous :/
	Real largest = input.Data.maxCoeff();

	Real sum = input.Data.unaryExpr(
		[largest](Real coeff)
		{
			return exp(coeff - largest);
		}).sum();

	Real sumDiv = 1.0 / sum;

	Vector ret = input.Data.unaryExpr(
		[largest, sumDiv](Real coeff)
		{
			return exp(coeff - largest) * sumDiv;
		});

#if _DEBUG
	// Verify the softmax invariant that the sum of the outputs sums to 1
	Real sftSum = ret.sum();
	assert(abs(1 - sftSum) < 0.00001);
#endif

	return Params(input, move(ret));
}

Params SoftmaxLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput,
	const Params &outputErrors)
{
	EstablishContext();

	// If the cost function is negative log loss, then it already computed
	// the gradient of it times the gradient of this layer, so just pass it
	// through
	if (_costIsLogLoss)
		return Params(lastInput, outputErrors.Data);

	Matrix m(lastOutput.Data.size(), lastOutput.Data.size());

	for (int y = 0; y < m.outerSize(); ++y)
	{
		for (int x = 0; x < m.innerSize(); ++x)
		{
			m(y, x) = lastOutput.Data(y) * ((x == y) - lastOutput.Data(x));
		}
	}

	Params inputErrors(lastInput, m * outputErrors.Data);

	return move(inputErrors);
}

void SoftmaxLayer::EstablishContext()
{
	if (_checked)
		return;
	_checked = true;

	if (!_net)
	{
		_costIsLogLoss = false;
		return;
	}

	ICost::Ptr cost = _net->GetCostFn();

	auto ll = dynamic_cast<LogLossCost*>(cost.get());

	_costIsLogLoss = ll != nullptr;
}

void BindStruct(const CStructBinder &binder, SoftmaxLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, SoftmaxLayer, SoftmaxLayer);


