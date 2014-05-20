#include "maxpool_layer.h"

#include <immintrin.h>

#include "thread/parallel_for.h"

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

	int miniBatchSize = (int)ceil(batchSize / float(s_threadPool.NumThreads()));

	//for (int imgIdx = 0; imgIdx < batchSize; ++imgIdx)
	FastFor(s_threadPool, 0, batchSize, miniBatchSize,
	    [&] (int baseImgIdx)
	{
	    for (int imgIdx = baseImgIdx, end = min(imgIdx + miniBatchSize, batchSize);
	         imgIdx < end; ++imgIdx)
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
	}
	);

	return move(output);
}

Params MaxPoolLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	const int ipWidth = lastInput.Width;
	const int ipHeight = lastInput.Height;
	const int depth = lastInput.Depth;
	const int batchSize = lastInput.BatchSize();

	const int ipStride = ipWidth * depth;

	const int opWidth = lastOutput.Width;
	const int opHeight = lastOutput.Height;

	Params inputErrors(ipWidth, ipHeight, depth,
			 CMatrix::Zero(lastInput.Data.rows(), lastInput.Data.cols()));

	const int opStride = opWidth * depth;

	int miniBatchSize = (int)ceil(batchSize / float(s_threadPool.NumThreads()));

	//for (int imgIdx = 0; imgIdx < batchSize; ++imgIdx)
    FastFor(s_threadPool, 0, batchSize, miniBatchSize,
        [&] (int baseImgIdx)
    {
        for (int imgIdx = baseImgIdx, end = min(imgIdx + miniBatchSize, batchSize);
             imgIdx < end; ++imgIdx)
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

                        const Real ipVal = lastInput.Data(inputIdx, imgIdx);
                        const Real opVal = lastOutput.Data(opIdx, imgIdx);

                        // If this value was the maximum, then backprop
                        // the error
                        if (ipVal == opVal)
                        {
                            Real &ipErrVal = inputErrors.Data(inputIdx, imgIdx);
                            const Real opErrVal = outputErrors.Data(opIdx, imgIdx);

                            ipErrVal = opErrVal;
                        }
                    }
                }
            }
        }
	}
    );

	return move(inputErrors);
}

void BindStruct(const CStructBinder &binder, MaxPoolLayer &layer)
{
	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MaxPoolLayer, MaxPoolLayer);

