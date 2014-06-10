#include "dropout_layer.h"

using namespace std;
using namespace axon::serialization;

DropoutLayer::DropoutLayer(string name, Real dropout)
	: SingleInputLayer(move(name)), _dropout(dropout)
{
}

Params DropoutLayer::SCompute(const Params &input, bool isTraining)
{
	Params output(input, new CMatrix(input.Rows, input.Cols));

	const CMatrix &mInput = input.GetHostMatrix();
	CMatrix &mOutput = output.GetHostMatrix();

	if (isTraining)
	{
		mOutput.noalias() = mInput.unaryExpr(
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
		mOutput.noalias() = mInput * (1.0f - _dropout);
	}

	return move(output);
}

Params DropoutLayer::SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params inputErrors(lastInput, new CMatrix(lastInput.Rows, lastInput.Cols));

	const CMatrix &mLastInput = lastInput.GetHostMatrix();
	const CMatrix &mLastOutput = lastOutput.GetHostMatrix();
	const CMatrix &mOutputErrors = outputErrors.GetHostMatrix();
	CMatrix &mInputErrors = inputErrors.GetHostMatrix();

	const Real *pLastInput = mLastInput.data(),
			   *pLastOutput = mLastOutput.data(),
			   *pOutputErrs = mOutputErrors.data();
	Real *pInputErrs = mInputErrors.data();
	Real *pEnd = pInputErrs + mInputErrors.size();

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

	BindStruct(binder, (SingleInputLayer&)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, DropoutLayer, DropoutLayer);
