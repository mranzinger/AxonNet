#include "maxpool_layer.h"

#include <immintrin.h>

#include "thread/parallel_for.h"

using namespace std;
using namespace axon::serialization;

MaxPoolLayer::MaxPoolLayer(string name, size_t windowSizeX, size_t windowSizeY)
	: SingleInputLayer(move(name)), _windowSizeX(windowSizeX), _windowSizeY(windowSizeY),
	  _stepX(0), _stepY(0)
{
}

Params MaxPoolLayer::SCompute(const Params &input, bool isTraining)
{
	if (_cuImpl)
		return _cuImpl->Compute(input);

	const int ipWidth = input.Width;
	const int ipHeight = input.Height;
	const int depth = input.Depth;
	const int batchSize = input.Cols;

	const int ipStride = ipWidth * depth;

	const int opWidth = (int) ceil(ipWidth / float(_windowSizeX));
	const int opHeight = (int) ceil(ipHeight / float(_windowSizeY));

	Params output(opWidth, opHeight, depth,
			new CMatrix(opWidth * opHeight * depth, batchSize));

	const CMatrix &mInput = input.GetHostMatrix();
	CMatrix &mOutput = output.GetHostMatrix();

	mOutput.setConstant(numeric_limits<Real>::lowest());

	const int opStride = opWidth * depth;

	int miniBatchSize = (int)ceil(batchSize / float(GetThreadPool().NumThreads()));

	//for (int imgIdx = 0; imgIdx < batchSize; ++imgIdx)
	FastFor(GetThreadPool(), 0, batchSize, miniBatchSize,
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

                        const Real ipVal = mInput(inputIdx, imgIdx);
                        Real &opVal = mOutput(opIdx, imgIdx);

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

Params MaxPoolLayer::SBackprop(const Params &lastInput, const Params &lastOutput,
                             const Params &outputErrors)
{
	if (_cuImpl)
		return _cuImpl->Backprop(lastInput, lastOutput, outputErrors);

	const int ipWidth = lastInput.Width;
	const int ipHeight = lastInput.Height;
	const int depth = lastInput.Depth;
	const int batchSize = lastInput.Cols;

	const int ipStride = ipWidth * depth;

	const int opWidth = lastOutput.Width;
	const int opHeight = lastOutput.Height;

	Params inputErrors(ipWidth, ipHeight, depth,
			 new CMatrix(lastInput.Rows, lastInput.Cols));

	const CMatrix &mLastInput = lastInput.GetHostMatrix();
	const CMatrix &mLastOutput = lastOutput.GetHostMatrix();
	const CMatrix &mOutputErrors = outputErrors.GetHostMatrix();

	CMatrix &mInputErrors = inputErrors.GetHostMatrix();
	mInputErrors.setZero();

	const int opStride = opWidth * depth;

	int miniBatchSize = (int)ceil(batchSize / float(GetThreadPool().NumThreads()));

	//for (int imgIdx = 0; imgIdx < batchSize; ++imgIdx)
    FastFor(GetThreadPool(), 0, batchSize, miniBatchSize,
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

                        const Real ipVal = mLastInput(inputIdx, imgIdx);
                        const Real opVal = mLastOutput(opIdx, imgIdx);

                        // If this value was the maximum, then backprop
                        // the error
                        if (ipVal == opVal)
                        {
                            Real &ipErrVal = mInputErrors(inputIdx, imgIdx);
                            const Real opErrVal = mOutputErrors(opIdx, imgIdx);

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

void MaxPoolLayer::OnInitCudaDevice(int deviceId)
{
	_cuImpl.reset(new CuMaxPoolLayer(deviceId, _windowSizeX, _windowSizeY,
									 _stepX, _stepY));
}

void BindStruct(const CStructBinder &binder, MaxPoolLayer &layer)
{
	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("stepX", layer._stepX)
		  ("stepY", layer._stepY);

	BindStruct(binder, (SingleInputLayer &)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MaxPoolLayer, MaxPoolLayer);


