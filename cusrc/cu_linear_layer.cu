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

public:
	Impl(int deviceId)
	{
		_handle = CuSetupProvider::GetHandle(deviceId);

		_weights.SetHandle(_handle);
	}

	Params Compute(const Params &input) const;
	Params Backprop(const Params &lastInput, const Params &lastOutput,
					const Params &outputErrors);

	void ApplyGradient();

	void SyncToDevice(const CWeights &hWeights);
	void SyncToHost(CWeights &hWeights) const;
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

void CuLinearLayer::SyncToDevice(const CWeights& hWeights)
{
	_impl->SyncToDevice(hWeights);
}

void CuLinearLayer::SyncToHost(CWeights& hWeights) const
{
	_impl->SyncToHost(hWeights);
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

Params CuLinearLayer::Impl::Compute(const Params& input) const
{
	const uint32_t batchSize = input.Cols;
	const uint32_t outputSize = _weights.Biases.Rows();
	const uint32_t inputSize = input.Rows;

	Params output(outputSize, 1, 1,
				new CuMat(_handle, outputSize, batchSize));

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
	Params inputErrors = Params::CreateLike(lastInput, _handle);

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

void CuLinearLayer::Impl::SyncToDevice(const CWeights& hWeights)
{
	_weights.CopyToDevice(hWeights);
}

void CuLinearLayer::Impl::SyncToHost(CWeights& hWeights) const
{
	_weights.CopyToHost(hWeights);
}
