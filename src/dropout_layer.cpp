#include "dropout_layer.h"

using namespace std;
using namespace axon::serialization;

DropoutLayer::DropoutLayer(string name, Real dropout)
	: SingleInputLayer(move(name)), _dropout(dropout)
{
}

Params DropoutLayer::SCompute(const Params &input, bool isTraining)
{
	Params ret(input, CMatrix());
	ret.Data.resize(input.Data.rows(), input.Data.cols());

	if (isTraining)
	{
		ret.Data.noalias() = input.Data.unaryExpr(
			[this] (Real val)
			{
				Real rndVal = _rand.Next();

				if (rndVal < _dropout)
					return 0.0f;
				else
					return val;
			}
		);
	}
	else
	{
		ret.Data.noalias() = input.Data * (1.0f - _dropout);
	}

	return move(ret);
}

Params DropoutLayer::SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params inputErrors(lastInput, CMatrix());
	inputErrors.Data.resize(lastInput.Data.rows(), lastInput.Data.cols());

	const Real *pLastInput = lastInput.Data.data(),
			   *pLastOutput = lastOutput.Data.data(),
			   *pOutputErrs = outputErrors.Data.data();
	Real *pInputErrs = inputErrors.Data.data();
	Real *pEnd = pInputErrs + inputErrors.Data.size();

	for (; pInputErrs != pEnd; ++pInputErrs, ++pLastInput, ++pLastOutput, ++pOutputErrs)
	{
		if (*pLastInput == *pLastOutput)
			*pInputErrs = *pOutputErrs;
		else
			*pInputErrs = 0.0f;
	}

	return move(inputErrors);
}

void BindStruct(const CStructBinder &binder, DropoutLayer &layer)
{
	binder("dropout", layer._dropout);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, DropoutLayer, DropoutLayer);
