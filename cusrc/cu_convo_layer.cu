/*
 * cu_convo_layer.cu
 *
 *  Created on: Jun 15, 2014
 *      Author: mike
 */

#include "cu_convo_layer.cuh"

#include "cusetup_provider.cuh"
#include "cu_weights.cuh"
#include "cumath_functions.cuh"

using namespace std;

class CuConvoLayer::Impl
{
private:
	CuContext _handle;

	CuWeights _weights;

	CuMat _cacheCompute;
	CuMat _cacheBackprop;

	int _windowSizeX, _windowSizeY;
	int _padWidth, _padHeight;
	int _strideX, _strideY;

public:
	Impl(int deviceId,
		 int windowSizeX, int windowSizeY,
		 int strideX, int strideY,
		 int padWidth, int padHeight)
		: _windowSizeX(windowSizeX), _windowSizeY(windowSizeY),
		  _padWidth(padWidth), _padHeight(padHeight),
		  _strideX(strideX), _strideY(strideY)
	{
		_handle = CuSetupProvider::GetHandle(deviceId);

		_weights.SetHandle(_handle);

		_cacheCompute.SetHandle(_handle);
		_cacheCompute.SetSharedModify(true);

		_cacheBackprop.SetHandle(_handle);
		_cacheBackprop.SetSharedModify(true);
	}
	~Impl()
	{
	}

	Params Compute(const Params &input);
	Params Backprop(const Params &lastInput, const Params &lastOutput,
					const Params &outputErrors);

	void ApplyGradient();

	void SyncToDevice(const CWeights &hWeights, bool gradToo);
	void SyncToHost(CWeights &hWeights, bool gradToo) const;

	void SetLearningRate(Real rate);
	void SetMomentum(Real rate);
	void SetWeightDecay(Real rate);
};

CuConvoLayer::CuConvoLayer(int deviceId,
						   int windowSizeX, int windowSizeY,
						   int strideX, int strideY,
						   int padWidth, int padHeight)
{
	_impl = new Impl(deviceId,
					 windowSizeX, windowSizeY,
					 strideX, strideY,
					 padWidth, padHeight);
}

CuConvoLayer::~CuConvoLayer()
{
	delete _impl;
}

Params CuConvoLayer::Compute(const Params& input) const
{
    return _impl->Compute(input);
}

Params CuConvoLayer::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors)
{
    return _impl->Backprop(lastInput, lastOutput, outputErrors);
}

void CuConvoLayer::ApplyGradient()
{
    _impl->ApplyGradient();
}

void CuConvoLayer::SyncToDevice(const CWeights& hWeights, bool gradToo)
{
    _impl->SyncToDevice(hWeights, gradToo);
}

void CuConvoLayer::SyncToHost(CWeights& hWeights, bool gradToo) const
{
    _impl->SyncToHost(hWeights, gradToo);
}

void CuConvoLayer::SetLearningRate(Real rate)
{
    _impl->SetLearningRate(rate);
}

void CuConvoLayer::SetMomentum(Real rate)
{
    _impl->SetMomentum(rate);
}



void CuConvoLayer::SetWeightDecay(Real rate)
{
    _impl->SetWeightDecay(rate);
}

__global__ void CuConvoLayer_Compute(const Real *gInput, Real *gOutput,
									 const Real *gWeights, const Real *gBiases,
									 const int numLayers,
									 const int ipWidth, const int ipHeight, const int ipDepth,
									 const int opWidth, const int opHeight, const int opDepth,
									 const int wndSizeX, const int wndSizeY,
									 const int strideX, const int strideY,
									 const int padWidth, const int padHeight)
{
	int destX = blockIdx.x * blockDim.x + threadIdx.x;
	int destY = blockIdx.y * blockDim.y + threadIdx.y;

	if (destX >= opWidth || destY >= opHeight)
	    return;

	const int layer = blockIdx.z;

	const Real *lInput = gInput + layer * (ipWidth * ipHeight * ipDepth);

	Real *lOutput = gOutput + layer * (opWidth * opHeight * opDepth);

	int srcX = -padWidth + destX * strideX;
	int srcY = -padHeight + destY * strideY;

	int xMin = max(0, srcX);
	int yMin = max(0, srcY);

	int xMax = min(srcX + wndSizeX, ipWidth);
	int yMax = min(srcY + wndSizeY, ipHeight);

	int kSkipX = xMin - srcX;
	int kSkipY = yMin - srcY;

	int iStride = ipWidth * ipDepth;
	int kStride = wndSizeX * ipDepth;

	int kfSkipStride = (kSkipY * kStride + kSkipX) * opDepth;
	int kInnerSkipStride = (kSkipX + (srcX + wndSizeX - xMax)) * opDepth;

	int xEnd = xMax * ipDepth;

	const int dxMin = xMin * ipDepth;

	const int opStoreIdx = destY * opWidth * opDepth + destX * opDepth;

	for (int dIdx = threadIdx.z; dIdx < opDepth; dIdx += blockDim.z)
	{
		const Real *lWeights = gWeights + dIdx;

		Real sum = gBiases[dIdx];

		const Real *iBuff = lInput + yMin * iStride;
		const Real *kBuff = lWeights + kfSkipStride;

		for (int iY = yMin; iY < yMax; ++iY, iBuff += iStride)
		{
			for (int iX = dxMin; iX < xEnd; ++iX, kBuff += opDepth)
			{
				const Real iVal = iBuff[iX];
				const Real kVal = kBuff[0];

				const Real product = iVal * kVal;

				sum += product;
			}

			// Skip over the padding parts of the filter
			kBuff += kInnerSkipStride;
		}

		// Finally, store the sum
		lOutput[opStoreIdx + dIdx] = sum;
	}
}


Params CuConvoLayer::Impl::Compute(const Params& input)
{
    const CuMat &mInput = input.GetCudaMatrix(_handle);

    const int ipWidth = input.Width;
    const int ipHeight = input.Height;
	const int ipDepth = input.Depth;
	const int batchSize = input.Cols;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (int) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (int) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opDepth = _weights.Weights.Rows();

	_cacheCompute.Resize(opWidth * opHeight * opDepth, batchSize);
	Params output(opWidth, opHeight, opDepth,
	            new CuMat(_cacheCompute));

	CuMat &mOutput = output.GetCudaMatrix(_handle);

	uint32_t blockDepth = min(opDepth, 64);

	dim3 blockSize(1, 1, blockDepth);
	dim3 gridSize(opWidth, opHeight, batchSize);

	cudaError_t err = cudaSetDevice(_handle.Device);

	if (err != cudaSuccess)
	    throw runtime_error("Unable to set the device.");

	CuConvoLayer_Compute
#ifdef _CUDA_COMPILE_
	    <<<gridSize, blockSize>>>
#endif
	                    (mInput.Buff(), mOutput.Buff(),
	                     _weights.Weights.Buff(), _weights.Biases.Buff(),
	                     batchSize,
	                     ipWidth, ipHeight, ipDepth,
	                     opWidth, opHeight, opDepth,
	                     _windowSizeX, _windowSizeY,
	                     _strideX, _strideY,
	                     _padWidth, _padHeight);

	err = cudaGetLastError();

	if (err != cudaSuccess)
		throw runtime_error("Unable to compute convolution.");

	return output;
}

Params CuConvoLayer::Impl::Backprop(const Params& lastInput,
        const Params& lastOutput, const Params& outputErrors)
{
    return lastInput;
}

void CuConvoLayer::Impl::ApplyGradient()
{
    _weights.ApplyGradient();
}

void CuConvoLayer::Impl::SyncToDevice(const CWeights& hWeights, bool gradToo)
{
    _weights.CopyToDevice(hWeights, gradToo);
}

void CuConvoLayer::Impl::SyncToHost(CWeights& hWeights, bool gradToo) const
{
    _weights.CopyToHost(hWeights, gradToo);
}

void CuConvoLayer::Impl::SetLearningRate(Real rate)
{
    _weights.LearningRate = rate;
}

void CuConvoLayer::Impl::SetMomentum(Real rate)
{
    _weights.Momentum = rate;
}

void CuConvoLayer::Impl::SetWeightDecay(Real rate)
{
    _weights.WeightDecay = rate;
}
