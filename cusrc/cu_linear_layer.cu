/*
 * cu_linear_layer.cu
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */

#include "cu_linear_layer.cuh"

#include "cusetup_provider.cuh"
#include "cu_weights.cuh"

class CuLinearLayer::Impl
{
private:
	CuContext _handle;

	CuWeights _weights;

	CuMat _computeCache;
	CuMat _backpropCache;

public:
	Impl(int deviceId)
	{
		_handle = CuSetupProvider::GetHandle(deviceId);

		_weights.SetHandle(_handle);
		_computeCache.SetHandle(_handle);
		_computeCache.SetSharedModify(true);
		_backpropCache.SetHandle(_handle);
		_backpropCache.SetSharedModify(true);
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


CuLinearLayer::CuLinearLayer(int deviceId)
{
	_impl = new Impl(deviceId);
}

CuLinearLayer::~CuLinearLayer()
{
	delete _impl;
}

Params CuLinearLayer::Compute(const Params& input) const
{
	return _impl->Compute(input);
}

Params CuLinearLayer::Backprop(const Params& lastInput, const Params& lastOutput,
		const Params& outputErrors)
{
	return _impl->Backprop(lastInput, lastOutput, outputErrors);
}

void CuLinearLayer::ApplyGradient()
{
	_impl->ApplyGradient();
}

void CuLinearLayer::SyncToDevice(const CWeights& hWeights, bool gradToo)
{
	_impl->SyncToDevice(hWeights, gradToo);
}

void CuLinearLayer::SyncToHost(CWeights& hWeights, bool gradToo) const
{
	_impl->SyncToHost(hWeights, gradToo);
}

void CuLinearLayer::SetLearningRate(Real rate)
{
	_impl->SetLearningRate(rate);
}

void CuLinearLayer::SetMomentum(Real rate)
{
	_impl->SetMomentum(rate);
}

void CuLinearLayer::SetWeightDecay(Real rate)
{
	_impl->SetWeightDecay(rate);
}

struct CuAddBias
{
	const Real *_bias;

	CuAddBias(const CuMat &biases)
		: _bias(biases.Buff()) { }

	__device__ Real operator()(Real val, uint32_t row, uint32_t col) const
	{
		return val + _bias[row];
	}
};

Params CuLinearLayer::Impl::Compute(const Params& input)
{
	const uint32_t batchSize = input.Cols;
	const uint32_t outputSize = _weights.Biases.Rows();
	const uint32_t inputSize = input.Rows;

	_computeCache.Resize(outputSize, batchSize);

	Params output(outputSize, 1, 1,
				new CuMat(_computeCache));

	const CuMat &mInput = input.GetCudaMatrix(_handle);
	CuMat &mOutput = output.GetCudaMatrix(_handle);

	// Compute the product of the weights and the input
	ScaledMultiply(1.0f, _weights.Weights, mInput, 0.0f, mOutput);

	// Add the biases to the output
	mOutput.UnaryExpr(CuAddBias(_weights.Biases));

	return output;
}

Params CuLinearLayer::Impl::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors)
{
	_backpropCache.Resize(lastInput.Rows, lastInput.Cols);

	Params inputErrors(lastInput, new CuMat(_backpropCache));

	const CuMat &mInput = lastInput.GetCudaMatrix(_handle);
	const CuMat &mOutputErrors = outputErrors.GetCudaMatrix(_handle);

	CuMat &mInputErrors = inputErrors.GetCudaMatrix(_handle);

	// Calculate the input error
	ScaledMultiply(1.0f, _weights.Weights.WeakTranspose(), mOutputErrors,
				   0.0f, mInputErrors);

	ScaledMultiply(1.0f, mOutputErrors, mInput.WeakTranspose(),
				   0.0f, _weights.WeightsGrad);

	_weights.BiasGrad = mOutputErrors.Rowwise().Sum();

	return inputErrors;
}

void CuLinearLayer::Impl::ApplyGradient()
{
	_weights.ApplyGradient();
}

void CuLinearLayer::Impl::SyncToDevice(const CWeights& hWeights, bool gradToo)
{
	_weights.CopyToDevice(hWeights, gradToo);
}

void CuLinearLayer::Impl::SyncToHost(CWeights& hWeights, bool gradToo) const
{
	_weights.CopyToHost(hWeights, gradToo);
}

void CuLinearLayer::Impl::SetLearningRate(Real rate)
{
	_weights.LearningRate = rate;
}

void CuLinearLayer::Impl::SetMomentum(Real rate)
{
	_weights.Momentum = rate;
}

void CuLinearLayer::Impl::SetWeightDecay(Real rate)
{
	_weights.WeightDecay = rate;
}
