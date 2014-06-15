/*
 * cu_convo_layer.cu
 *
 *  Created on: Jun 15, 2014
 *      Author: mike
 */

#include "cu_convo_layer.cuh"

#include "cusetup_provider.cuh"
#include "cu_weights.cuh"

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
									 const Real *gWeights,
									 const int ipWidth, const int ipHeight, const int ipDepth,
									 const int opWidth, const int opHeight, const int opDepth,
									 const int wndSizeX, const int wndSizeY,
									 const int strideX, const int strideY,
									 const int padWidth, const int padHeight)
{
	uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int dIdx = threadIdx.z;
	const int layer = blockIdx.z;

	const Real *lInput = gInput + layer * (ipWidth * ipHeight * ipDepth);
	const Real *pWeights = gWeights + dIdx * (wndSizeX * wndSizeY * ipDepth);

	Real *lOutput = gOutput + (opWidth * opHeight * opDepth);
}


Params CuConvoLayer::Impl::Compute(const Params& input)
{
    const CuMat &mInput = input.GetCudaMatrix(_handle);

    const int ipWidth = input.Width;
    const int ipHeight = input.Height;
	const int ipDepth = input.Depth;
	const int batchSize = input.Cols;

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (int) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (int) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opDepth = _weights.Weights.Rows();
	const int opStride = opWidth * opDepth;
}

Params CuConvoLayer::Impl::Backprop(const Params& lastInput,
        const Params& lastOutput, const Params& outputErrors)
{
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
