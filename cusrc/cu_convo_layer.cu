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



__shared__ extern Real sInput[];

template<int wndProcX>
__device__ void CuConvoLayer_Compute_Device(
                                 const Real *gInput, Real *gOutput,
                                 const Real *gWeights, const Real *gBiases,
                                 const int ipWidth, const int ipHeight, const int ipDepth,
                                 const int opWidth, const int opHeight, const int opDepth,
                                 const int wndSizeX, const int wndSizeY,
                                 const int strideX, const int strideY,
                                 const int padWidth, const int padHeight)
{
    // Switching the x's and the z's here
    int destX = blockIdx.z * blockDim.z + threadIdx.z;

    int destY = blockIdx.y * blockDim.y + threadIdx.y;

    int destZ = blockIdx.x * blockDim.x + threadIdx.x;

    int layer = destZ / opDepth;

    int dIdx = destZ % opDepth;

    const Real *lInput = gInput + layer * (ipWidth * ipHeight * ipDepth);

    Real *lOutput = gOutput + layer * (opWidth * opHeight * opDepth);

    int srcX = -padWidth + destX * strideX;
    int srcY = -padHeight + destY * strideY;

    int xMin = max(0, srcX);
    int yMin = max(0, srcY);

    int xMax = min(srcX + wndSizeX, ipWidth);
    int yMax = min(srcY + wndSizeY, ipHeight);

    int iStride = ipWidth * ipDepth;

    int xEnd = xMax * ipDepth;

    const int dxMin = xMin * ipDepth;

    const int endImgIdx = yMax * iStride;

    const int procInputWidth = xEnd - dxMin;

    /// !!!! Load the image buffer into shared memory !!!!
    // Calculate the number of warps that are in this block.
    // For coalesced access rules, we want these guys to be grouped on a row
    int numWarps = blockDim.x / 32;

    // Not enough threads to even fill a single warp...
    // This will not be ultra-efficient
    if (numWarps <= 1)
    {
        int startCol = dxMin + threadIdx.x;

        for (int iY = 0, imgIdx = yMin * iStride;
                imgIdx < endImgIdx;
                ++iY, imgIdx += iStride)
        {
            for (int iX = startCol; iX < xEnd; iX += blockDim.x)
            {
                const Real iVal = lInput[imgIdx + iX];

                sInput[iY * procInputWidth + iX - dxMin] = iVal;
            }
        }
    }
    else
    {
        int warpsPerRow = round_up(numWarps, yMax - yMin);
        int simulRows = numWarps / warpsPerRow;

        // Let each warp do a separate row
        int startRow = threadIdx.x / (32 * warpsPerRow);
        int startCol = dxMin + (threadIdx.x % (32 * warpsPerRow));

        for (int iY = startRow, imgIdx = (yMin + startRow) * iStride;
                 imgIdx < endImgIdx;
                 iY += simulRows, imgIdx += (simulRows * iStride))
        {
            for (int iX = startCol; iX < xEnd; iX += (32 * warpsPerRow))
            {
                const Real iVal = lInput[imgIdx + iX];

                sInput[iY * procInputWidth + iX - dxMin] = iVal;
            }
        }
    }

    __syncthreads();

    int kSkipX = xMin - srcX;
    int kSkipY = yMin - srcY;

    int kStride = wndSizeX * ipDepth;

    int kfSkipStride = (kSkipY * kStride + kSkipX * ipDepth) * opDepth;
    int kInnerSkipStride = (kSkipX + (srcX + wndSizeX - xMax)) * ipDepth * opDepth;

    int weightsIdx = dIdx + kfSkipStride;

    Real sum = gBiases[dIdx];

    int ipIdx = 0;
    for (int iY = yMin; iY < yMax; ++iY)
    {
        for (int iX = dxMin; iX < xEnd; iX += wndProcX)
        {
            #pragma unroll
            for (int iW = 0; iW < wndProcX; ++iW, ++ipIdx, weightsIdx += opDepth)
            {
                const Real iVal = sInput[ipIdx];
                const Real kVal = gWeights[weightsIdx];

                const Real product = iVal * kVal;

                sum += product;
            }
        }

        // Skip over the padding parts of the filter
        weightsIdx += kInnerSkipStride;
    }

    const int opStoreIdx = destY * opWidth * opDepth + destX * opDepth;

    // Finally, store the sum
    lOutput[opStoreIdx + dIdx] = sum;
}

__global__ void CuConvoLayer_Compute(const Real *gInput, Real *gOutput,
                                     const Real *gWeights, const Real *gBiases,
                                     //const int numLayers,
                                     const int ipWidth, const int ipHeight, const int ipDepth,
                                     const int opWidth, const int opHeight, const int opDepth,
                                     const int wndSizeX, const int wndSizeY,
                                     const int strideX, const int strideY,
                                     const int padWidth, const int padHeight)
{
    // Switching the x's and the z's here
    int destX = blockIdx.z * blockDim.z + threadIdx.z;

    if (destX >= opWidth)
        return;

    int srcX = -padWidth + destX * strideX;

    int xMin = max(0, srcX);

    int xMax = min(srcX + wndSizeX, ipWidth);

    const int kernWidth = xMax - xMin;

    if (kernWidth <= 0)
        return;

#define MAKE_CONVO_CALL(procWidth) \
    case procWidth: \
            CuConvoLayer_Compute_Device<procWidth>( \
                                         gInput, gOutput, \
                                         gWeights, gBiases, \
                                         ipWidth, ipHeight, ipDepth, \
                                         opWidth, opHeight, opDepth, \
                                         wndSizeX, wndSizeY, \
                                         strideX, strideY, \
                                         padWidth, padHeight); \
    break



    switch (kernWidth)
    {
    MAKE_CONVO_CALL(1);
    MAKE_CONVO_CALL(2);
    MAKE_CONVO_CALL(3);
    MAKE_CONVO_CALL(4);
    MAKE_CONVO_CALL(5);
    MAKE_CONVO_CALL(6);
    MAKE_CONVO_CALL(7);
    MAKE_CONVO_CALL(8);
    MAKE_CONVO_CALL(9);
    MAKE_CONVO_CALL(10);
    MAKE_CONVO_CALL(11);
    MAKE_CONVO_CALL(12);
    MAKE_CONVO_CALL(13);
    MAKE_CONVO_CALL(14);
    MAKE_CONVO_CALL(15);
    MAKE_CONVO_CALL(16);
    MAKE_CONVO_CALL(17);
    MAKE_CONVO_CALL(18);
    MAKE_CONVO_CALL(19);
    MAKE_CONVO_CALL(20);
    MAKE_CONVO_CALL(21);
    }

#undef MAKE_CONVO_CALL
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

	uint32_t blockDepth = min(opDepth, 1024);

	dim3 blockSize(blockDepth, 1, 1);
	dim3 gridSize = round_up(opDepth * batchSize, opHeight, opWidth, blockSize);

	uint32_t smemSize = _windowSizeX * _windowSizeY * ipDepth * sizeof(Real);

	CuConvoLayer_Compute
        <<<gridSize, blockSize, smemSize>>>
                        (mInput.Buff(), mOutput.Buff(),
                         _weights.Weights.Buff(), _weights.Biases.Buff(),
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
