#include "dropout_layer.h"

using namespace std;
using namespace axon::serialization;

DropoutLayer::DropoutLayer(string name, Real dropout)
	: LayerBase(move(name)), _dropout(dropout)
{
	_trainGens.resize(1);
	_trainRands.resize(1);
}

Vector DropoutLayer::Compute(int threadIdx, const Vector &input, bool isTraining)
{
	Vector ret(input.size());

	if (isTraining)
	{
		Dropout(threadIdx, input, ret);
	}
	else
	{
		ret.noalias() = input * _dropout;
	}

	return move(ret);
}

Vector DropoutLayer::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors)
{
	Vector inputErrors(outputErrors.size());

	Dropout(threadIdx, outputErrors, inputErrors);

	return inputErrors;
}

void DropoutLayer::PrepareForThreads(size_t num)
{
	_trainRands.resize(max<size_t>(1, num));
	_trainGens.resize(max<size_t>(1, num));
}

void DropoutLayer::Dropout(int threadIdx, const Vector &input, Vector &output)
{
	RandVec &vec = _trainRands[threadIdx];
	DropRand &gen = _trainGens[threadIdx];

	vec.resize(input.size() / 64 + 1);

	// Generate a random distribution. 1-bit per input
	for (uint64_t &val : vec)
		val = gen.Next();

	Real *pOut = output.data();
	const Real *pIn = input.data();

	size_t numBatches = output.size() / 64;

	for (size_t i = 0; i < numBatches; ++i)
	{
		uint64_t rnd = vec[i];

		for (Real *pEnd = pOut + 64; pOut != pEnd; ++pOut, ++pIn, rnd >>= 1)
		{
			bool act = rnd & 0x1;

			if (act)
				*pOut = *pIn;
			else
				*pOut = 0;
		}
	}

	uint64_t rnd = vec[numBatches];

	for (Real *pEnd = output.data() + output.size(); pOut != pEnd; ++pOut, ++pIn, rnd >>= 1)
	{
		bool act = rnd & 0x1;

		if (act)
			*pOut = *pIn;
		else
			*pOut = 0;
	}
}

void BindStruct(const CStructBinder &binder, DropoutLayer &layer)
{
	binder("dropout", layer._dropout);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, DropoutLayer, DropoutLayer);