#include "softmax_layer.h"
#include "functions.h"

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
	/*Vector v = ApplyDerivative<LogisticFn>(lastOutput);
	v = v.binaryExpr(outputErrors, [](Real a, Real b) { return a * b; });
	return v;*/
	return Params(lastInput, outputErrors.Data);
}

void BindStruct(const CStructBinder &binder, SoftmaxLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, SoftmaxLayer, SoftmaxLayer);