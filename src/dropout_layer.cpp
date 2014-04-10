#include "dropout_layer.h"

using namespace std;
using namespace axon::serialization;

DropoutLayer::DropoutLayer(string name, Real dropout)
	: LayerBase(move(name)), _dropout(dropout)
{
	_trainGens.resize(1);
	_trainRands.resize(1);
}

Params DropoutLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	Params ret(input, CMatrix());
	ret.Data.resize(input.Data.rows(), input.Data.cols());

	if (isTraining)
	{
		Dropout(threadIdx, input.Data, ret.Data, true);
	}
	else
	{
		ret.Data.noalias() = input.Data * _dropout;
	}

	return move(ret);
}

Params DropoutLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params inputErrors(lastInput, CMatrix());
	inputErrors.Data.resize(lastInput.Data.rows(), lastInput.Data.cols());

	Dropout(threadIdx, outputErrors.Data, inputErrors.Data, false);

	return move(inputErrors);
}

void DropoutLayer::PrepareForThreads(size_t num)
{
	_trainRands.resize(max<size_t>(1, num));
	_trainGens.resize(max<size_t>(1, num));
}

void DropoutLayer::Dropout(int threadIdx, const CMatrix &input, CMatrix &output, bool generate)
{
	RandVec &vec = _trainRands[threadIdx];
	DropRand &gen = _trainGens[threadIdx];

	if (generate)
	{
		vec.resize(input.size() / 64 + 1);

		// Generate a random distribution. 1-bit per input
		for (uint64_t &val : vec)
			val = gen.Next();
	}

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
