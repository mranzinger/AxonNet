#include "softmax_layer.h"
#include "functions.h"
#include "logloss_cost.h"
#include "neural_net.h"

using namespace std;
using namespace axon::serialization;

SoftmaxLayer::SoftmaxLayer()
    : SoftmaxLayer("")
{
}
SoftmaxLayer::SoftmaxLayer(string name)
    : SoftmaxLayer(move(name), "")
{

}
SoftmaxLayer::SoftmaxLayer(string name, string inputName)
    : SingleInputLayer(move(name), move(inputName)),
      _checked(false), _costIsLogLoss(false)
{
}

Params SoftmaxLayer::SCompute(const Params &input, bool isTraining)
{
	// Getting the max value and shifting the dimension is a neat trick to prevent overflow
	// NOTE: Underflow may still occur though, but that is far less dangerous :/

	if (_cuImpl)
		return _cuImpl->Compute(input);

	Params ret(input, new CMatrix(input.Rows, input.Cols));

	CMatrix &mOutput = ret.GetHostMatrix();
	const CMatrix &mInput = input.GetHostMatrix();

	for (int col = 0, cEnd = input.Cols; col < cEnd; ++col)
	{
		auto vColInput = mInput.col(col);
		auto vColOutput = mOutput.col(col);

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

Params SoftmaxLayer::SBackprop(const Params &lastInput, const Params &lastOutput,
	const Params &outputErrors)
{
	EstablishContext();

	if (_cuImpl)
		return _cuImpl->Backprop(lastInput, lastOutput, outputErrors);

	// If the cost function is negative log loss, then it already computed
	// the gradient of it times the gradient of this layer, so just pass it
	// through
	if (_costIsLogLoss)
		return outputErrors;

	Params inputErrors = Params::CreateLike(lastInput);

	const CMatrix &mLastOp = lastOutput.GetHostMatrix();

	// Compute the full Jacobian for each of the output error vectors
	RMatrix m(lastOutput.Rows, lastOutput.Rows);

	for (int col = 0, cEnd = lastOutput.Cols; col < cEnd; ++col)
	{
		auto vLastOutput = mLastOp.col(col);

		for (int y = 0; y < m.outerSize(); ++y)
		{
			for (int x = 0; x < m.innerSize(); ++x)
			{
				m(y, x) = vLastOutput(y) * ((x == y) - vLastOutput(x));
			}
		}

		auto vOutputError = outputErrors.GetHostMatrix().col(col);
		auto vInputError = inputErrors.GetHostMatrix().col(col);

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

	if (_cuImpl)
		_cuImpl->SetCostIsLogreg(_costIsLogLoss);
}

void SoftmaxLayer::OnInitCudaDevice(int deviceId)
{
	_cuImpl.reset(new CuSoftmaxLayer(deviceId));
}

void BindStruct(const CStructBinder &binder, SoftmaxLayer &layer)
{
	BindStruct(binder, (SingleInputLayer&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, SoftmaxLayer, SoftmaxLayer);


