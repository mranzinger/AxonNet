#include "maxpool_layer.h"

using namespace std;
using namespace axon::serialization;

/*
using Eigen::MatrixXi;

MaxPoolLayer::MaxPoolLayer(string name, size_t windowSizeX, size_t windowSizeY)
	: LayerBase(move(name)), _windowSizeX(windowSizeX), _windowSizeY(windowSizeY)
{
	PrepareForThreads(1);
}

Params MaxPoolLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	const size_t opWidth = (size_t)ceil(float(input.Width) / _windowSizeX);
	const size_t opHeight = (size_t)ceil(float(input.Height) / _windowSizeY);

	const size_t depth = input.Depth;

	Vector vOutput(opWidth * opHeight * depth);
	vOutput.setConstant(numeric_limits<Real>::lowest());

	RMap mOutput(vOutput.data(), opHeight, opWidth * depth);
	RMap mInput(const_cast<Real*>(input.Data.data()), input.Height, input.Width * input.Depth);

	MatrixXi mIndexes(opHeight, opWidth * depth);

	const size_t inputStride = input.Width * input.Depth;

	for (size_t bucketY = 0; bucketY < opHeight; ++bucketY)
	{
		size_t startY = bucketY * _windowSizeY;

		for (size_t bucketX = 0; bucketX < opWidth; ++bucketX)
		{
			size_t startX = bucketX * _windowSizeX * depth;

			for (size_t y = 0; 
					y < _windowSizeY &&
					y < (input.Height - startY);
					++y)
			{
				size_t wndY = startY + y;

				for (size_t x = 0;
					 x < _windowSizeX &&
					 x < (inputStride - startX);
					 x += depth)
				{
					size_t wndX = startX + x;

					for (size_t d = 0; d < depth; ++d)
					{
						const Real ipVal = mInput(wndY, wndX + d);

						Real &opMax = mOutput(bucketY, bucketX * depth + d);

						if (ipVal > opMax)
						{
							opMax = ipVal;
							
							if (isTraining)
							{
								size_t idx = wndY * inputStride + wndX + d;

								mIndexes(bucketY, bucketX * depth + d) = idx;
							}
						}
					}
				}
			}
		}
	}

	if (isTraining)
	{
		_threadIndexes[threadIdx].swap(mIndexes);
	}

	Params ret(opWidth, opHeight, depth, Vector());
	ret.Data.swap(vOutput);

	return move(ret);
}

Params MaxPoolLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params ret(lastInput, Vector::Zero(lastInput.Width * lastInput.Height * lastInput.Depth));

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

	return move(ret);
}

void MaxPoolLayer::PrepareForThreads(size_t num)
{
	_threadIndexes.resize(num);
}

void BindStruct(const CStructBinder &binder, MaxPoolLayer &layer)
{
	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MaxPoolLayer, MaxPoolLayer);

*/
