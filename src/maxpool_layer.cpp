#include "maxpool_layer.h"

using namespace std;
using namespace axon::serialization;

MaxPoolLayer::MaxPoolLayer(string name, size_t windowSizeX, size_t windowSizeY)
	: LayerBase(move(name)), _windowSizeX(windowSizeX), _windowSizeY(windowSizeY)
{
	PrepareForThreads(1);
}

Params MaxPoolLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	const int ipWidth = input.Width;
	const int ipHeight = input.Height;
	const int depth = input.Depth;
	const int batchSize = input.BatchSize();

	const int ipStride = ipWidth * depth;

	const int opWidth = (int) ceil(ipWidth / float(_windowSizeX));
	const int opHeight = (int) ceil(ipHeight / float(_windowSizeY));

	Params output(opWidth, opHeight, depth,
			CMatrix(opWidth * opHeight * depth, batchSize),
			Params::Packed);
	output.Data.setConstant(numeric_limits<Real>::lowest());

	const int opStride = opWidth * depth;

	for (int imgIdx = 0; imgIdx < batchSize; ++imgIdx)
	{
		for (int y = 0; y < ipHeight; ++y)
		{
			int opY = y / _windowSizeY;

			for (int x = 0; x < ipWidth; ++x)
			{
				int opX = x / _windowSizeX;

				for (int c = 0; c < depth; ++c)
				{
					int inputIdx = y * ipStride + x * depth + c;
					int opIdx = opY * opStride + opX * depth + c;

					const Real ipVal = input.Data(inputIdx, imgIdx);
					Real &opVal = output.Data(opIdx, imgIdx);

					if (ipVal > opVal)
						opVal = ipVal;
				}
			}
		}
	}

	return move(output);
}

Params MaxPoolLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	/*Params ret(lastInput, Vector::Zero(lastInput.Width * lastInput.Height * lastInput.Depth));

	const MatrixXi &mIndexes = _threadIndexes[threadIdx];

	const int *pIdx = mIndexes.data();
	const int *pEnd = pIdx + mIndexes.size();

	const Real *pOpErrs = outputErrors.Data.data();
	Real *pIpErrs = ret.Data.data();

	// Only the input that produced the maximum value propagates errors
	for (; pIdx != pEnd; ++pIdx, ++pOpErrs)
	{
		pIpErrs[*pIdx] = *pOpErrs;
	}

	return move(ret);*/
	throw runtime_error("Not implemented");
}

void BindStruct(const CStructBinder &binder, MaxPoolLayer &layer)
{
	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MaxPoolLayer, MaxPoolLayer);

