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

	Params ret(input, CMatrix(input.Data.rows(), input.Data.cols()));

	for (int col = 0, cEnd = input.Data.cols(); col < cEnd; ++col)
	{
		auto vColInput = input.Data.col(col);
		auto vColOutput = ret.Data.col(col);

		Real largest = vColInput.maxCoeff();

		Real sum = vColInput.unaryExpr(
			[largest](Real coeff)
			{
				return exp(coeff - largest);
			}).sum();

		Real sumDiv = 1.0 / sum;

		vColOutput = vColInput.unaryExpr(
			[largest, sumDiv](Real coeff)
			{
				return exp(coeff - largest) * sumDiv;
			});

#if _DEBUG
		// Verify the softmax invariant that the sum of the outputs sums to 1
		Real sftSum = vColOutput.sum();
		assert(abs(1 - sftSum) < 0.00001);
#endif

	}

	return move(ret);
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

	Params inputErrors(lastInput, CMatrix(lastInput.Data.rows(), lastInput.Data.cols()));

	// Compute the full Jacobian for each of the output error vectors
	RMatrix m(lastOutput.Data.rows(), lastOutput.Data.rows());

	for (int col = 0, cEnd = lastOutput.Data.cols(); col < cEnd; ++col)
	{
		auto vLastOutput = lastOutput.Data.col(col);

		for (int y = 0; y < m.outerSize(); ++y)
		{
			for (int x = 0; x < m.innerSize(); ++x)
			{
				m(y, x) = vLastOutput(y) * ((x == y) - vLastOutput(x));
			}
		}

		auto vOutputError = outputErrors.Data.col(col);
		auto vInputError = inputErrors.Data.col(col);

		vInputError = m * vOutputError;
	}

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

	// The cost function must be log loss, and this has to be the final layer
	// in the network to be able to use the shortcut
	_costIsLogLoss = ll != nullptr &&
					 _net->GetLayer(_net->NumLayers() - 1).get() == this;
}

void BindStruct(const CStructBinder &binder, SoftmaxLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, SoftmaxLayer, SoftmaxLayer);


