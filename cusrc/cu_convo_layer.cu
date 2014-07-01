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

struct PlacementParams
{
    int KSkipStride;
    int KInnerSkipStride;
};

struct ConvoKernelParams
{
    int IpDepth;
    int OpDepth;
    int WindowSizeX;
    int WindowSizeY;
    int StrideX;
    int StrideY;
    int PadWidth;
    int PadHeight;

    Real *WeightsBuff;
    Real *BiasBuff;

    PlacementParams Places[20];
};

template<int numImagesPerThread>
__global__ void CuConvoLayer_Compute(const Real *gInput, Real *gOutput,
                                     const Real *gWeights, const Real *gBiases,
                                     //const int numLayers,
                                     const int ipWidth, const int ipHeight, const int ipDepth,
                                     const int opWidth, const int opHeight, const int opDepth,
                                     const int wndSizeX, const int wndSizeY,
                                     const int strideX, const int strideY,
                                     const int padWidth, const int padHeight)
{
	__shared__ extern Real sInput[];

	// Switching the x's and the z's here
    const int destX = blockIdx.z * blockDim.z + threadIdx.z;

    const int destY = blockIdx.y * blockDim.y + threadIdx.y;

    //if (destX >= opWidth || destY >= opHeight)
    //    return;

    //const int destZ = blockIdx.x * blockDim.x + threadIdx.x;

    //const int layer = destZ / opDepth;
    const int layer = blockIdx.x * numImagesPerThread;

    //if (layer >= numLayers)
    //    return;

    //const int dIdx = destZ % opDepth;
    const int dIdx = threadIdx.x;

    const int ipImgSize = ipWidth * ipHeight * ipDepth;

    const Real *lInput = gInput + layer * ipImgSize;

    const int opImgSize = opWidth * opHeight * opDepth;

    Real *lOutput = gOutput + layer * opImgSize;

    const int srcX = -padWidth + destX * strideX;
    const int srcY = -padHeight + destY * strideY;

    const int xMin = max(0, srcX);
    const int yMin = max(0, srcY);

    const int xMax = min(srcX + wndSizeX, ipWidth);
    const int yMax = min(srcY + wndSizeY, ipHeight);

    //const int wndProcX = xMax - xMin;

    const int kSkipX = xMin - srcX;
    const int kSkipY = yMin - srcY;

    const int iStride = ipWidth * ipDepth;
    const int kStride = wndSizeX * ipDepth;

    const int kfSkipStride = (kSkipY * kStride + kSkipX * ipDepth) * opDepth;
    const int kInnerSkipStride = (kSkipX + (srcX + wndSizeX - xMax)) * ipDepth * opDepth;

    const int xEnd = xMax * ipDepth;

    const int dxMin = xMin * ipDepth;

    //int imgIdx = yMin * iStride;
    int weightsIdx = dIdx + kfSkipStride;

    const int endImgIdx = yMax * iStride;

    const int procInputWidth = xEnd - dxMin;
    const int procInputSize = procInputWidth * (yMax - yMin);

    /// !!!! Load the image buffer into shared memory !!!!
    // Calculate the number of warps that are in this block.
    // For coalesced access rules, we want these guys to be grouped on a row
    const int numWarps = blockDim.x / 32;

    // Not enough threads to even fill a single warp...
    // This will not be ultra-efficient
    if (numWarps <= 1)
    {
        const int startCol = dxMin + threadIdx.x;

        for (int iY = 0, imgIdx = yMin * iStride;
                imgIdx < endImgIdx;
                ++iY, imgIdx += iStride)
        {
            for (int iX = startCol; iX < xEnd; iX += blockDim.x)
            {
				#pragma unroll
            	for (int k = 0; k < numImagesPerThread; ++k)
            	{
            		const Real iVal = lInput[imgIdx + (k * ipImgSize) + iX];

            		sInput[(k * procInputSize) + (iY * procInputWidth + iX - dxMin)] = iVal;
            	}
            }
        }
    }
    else
    {
        const int warpsPerRow = round_up(numWarps, yMax - yMin);
        const int simulRows = numWarps / warpsPerRow;

        // Let each warp do a separate row
        const int startRow = threadIdx.x / (32 * warpsPerRow);
        const int startCol = dxMin + (threadIdx.x % (32 * warpsPerRow));

        for (int iY = startRow, imgIdx = (yMin + startRow) * iStride;
                 imgIdx < endImgIdx;
                 iY += simulRows, imgIdx += (simulRows * iStride))
        {
            for (int iX = startCol; iX < xEnd; iX += (32 * warpsPerRow))
            {
				#pragma unroll
            	for (int k = 0; k < numImagesPerThread; ++k)
            	{
            		const Real iVal = lInput[imgIdx + (k * ipImgSize) + iX];

            		sInput[(k * procInputSize) + (iY * procInputWidth + iX - dxMin)] = iVal;
            	}
            }
        }
    }

    __syncthreads();

    //Real sum = gBiases[dIdx];
    Real sum[numImagesPerThread] = { 0.0f };

    // Peel vectors of 8
    const int vecProcX = procInputWidth & ~0x7;
    const int vecTailX = procInputWidth & 0x7;

    const int vecXend = dxMin + vecProcX;

    int ipIdx = 0;
    for (int iY = yMin; iY < yMax; ++iY)
    {
    	for (int iX = dxMin; iX < vecXend; iX += 8)
    	{
			#pragma unroll
    		for (int i = 0; i < 8; ++i)
    		{
    			const Real kVal = gWeights[weightsIdx + i * opDepth];

				#pragma unroll
    			for (int k = 0; k < numImagesPerThread; ++k)
    			{
    				const Real iVal = sInput[ipIdx + (k * procInputSize) + i];

    				const Real product = iVal * kVal;

    				sum[k] += product;
    			}
    		}

    		ipIdx += 8;
    		weightsIdx += 8 * opDepth;
    	}

#define DUFF_CASE(v) { \
			const Real kVal = gWeights[weightsIdx + (v - 1) * opDepth]; \
    		for (int k = 0; k < numImagesPerThread; ++k) \
    		{ \
    			sum[k] += sInput[ipIdx + (k * procInputSize) + (v - 1)] * kVal; \
    		} }

    	switch (vecTailX)
    	{
    	case 7:
		//#pragma unroll
    	DUFF_CASE(7);
    	case 6:
		//#pragma unroll
    	DUFF_CASE(6);
    	case 5:
		//#pragma unroll
    	DUFF_CASE(5);
    	case 4:
		//#pragma unroll
    	DUFF_CASE(4);
    	case 3:
		//#pragma unroll
    	DUFF_CASE(3);
    	case 2:
		//#pragma unroll
    	DUFF_CASE(2);
    	case 1:
		//#pragma unroll
    	DUFF_CASE(1);
    	case 0:
    		break;
    	}

    	ipIdx += vecTailX;
    	weightsIdx += vecTailX * opDepth;

#undef DUFF_CASE

        // Skip over the padding parts of the filter
        weightsIdx += kInnerSkipStride;
    }

    const int opStoreIdx = destY * opWidth * opDepth + destX * opDepth;

    // Finally, store the sum
    //lOutput[opStoreIdx + dIdx] = sum;
    const Real bias = gBiases[dIdx];
	#pragma unroll
    for (int k = 0; k < numImagesPerThread; ++k)
    {
    	lOutput[opStoreIdx + (k * opImgSize) + dIdx] = sum[k] + bias;
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

	cudaError_t err = cudaSetDevice(_handle.Device);

	if (err != cudaSuccess)
	    throw runtime_error("Unable to set the device.");

	if (opDepth > 1024)
		throw runtime_error("Output depths greater than 1024 are not supported.");

	uint32_t blockDepth = opDepth;

	dim3 blockSize(blockDepth, 1, 1);
	dim3 gridSize = round_up(blockDepth * batchSize, opHeight, opWidth, blockSize);

	uint32_t smemSize = _windowSizeX * _windowSizeY * ipDepth * sizeof(Real);

	uint32_t numImagesPerThread = 1;
	if ((batchSize % 4) == 0)
		numImagesPerThread = 4;
	else if ((batchSize % 3) == 0)
		numImagesPerThread = 3;
	else if ((batchSize % 2) == 0)
		numImagesPerThread = 2;

	smemSize *= numImagesPerThread;
	gridSize.x /= numImagesPerThread;

	//cudaFuncSetCacheConfig(CuConvoLayer_Compute, cudaFuncCachePreferShared);

#define LAUNCH_CONVO_KERNEL(v) \
			CuConvoLayer_Compute \
				<v> \
				<<<gridSize, blockSize, smemSize>>> \
					(mInput.Buff(), mOutput.Buff(), \
				     _weights.Weights.Buff(), _weights.Biases.Buff(), \
				     ipWidth, ipHeight, ipDepth, \
				     opWidth, opHeight, opDepth, \
				     _windowSizeX, _windowSizeY, \
				     _strideX, _strideY, \
				     _padWidth, _padHeight)

	switch (numImagesPerThread)
	{
	case 1:
		LAUNCH_CONVO_KERNEL(1);
		break;
	case 2:
		LAUNCH_CONVO_KERNEL(2);
		break;
	case 3:
		LAUNCH_CONVO_KERNEL(3);
		break;
	case 4:
		LAUNCH_CONVO_KERNEL(4);
		break;
	}

	/*CuConvoLayer_Compute
        <<<gridSize, blockSize, smemSize>>>
                        (mInput.Buff(), mOutput.Buff(),
                         _weights.Weights.Buff(), _weights.Biases.Buff(),
                         ipWidth, ipHeight, ipDepth,
                         opWidth, opHeight, opDepth,
                         _windowSizeX, _windowSizeY,
                         _strideX, _strideY,
                         _padWidth, _padHeight);*/

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
