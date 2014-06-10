/*
 * cu_linear_layer.cu
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */

#include "cu_linear_layer.cu"

#include "cusetup_provider.cuh"

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
					const Params &outputErrors) const;

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
		const Params& outputErrors) const
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

Params CuLinearLayer::Impl::Compute(const Params& input) const
{

}

Params CuLinearLayer::Impl::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors) const
{
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
