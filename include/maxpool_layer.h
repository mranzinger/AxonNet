#pragma once

#include "single_input_layer.h"

#include "cu_maxpool_layer.cuh"

class NEURAL_NET_API MaxPoolLayer
	: public SingleInputLayer
{
scope_private:
	uint32_t _windowSizeX, _windowSizeY;

	uint32_t _stepX, _stepY;

	std::unique_ptr<CuMaxPoolLayer> _cuImpl;

scope_public:
	typedef std::shared_ptr<MaxPoolLayer> Ptr;

	MaxPoolLayer() = default;
	MaxPoolLayer(std::string name, size_t windowSizeX, size_t windowSizeY);

	virtual std::string GetLayerType() const override { return "Max Pool Layer"; }

	friend void BindStruct(const aser::CStructBinder &binder, MaxPoolLayer &layer);

scope_protected:
    virtual Params SCompute(const Params &input, bool isTraining) override;
    virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

    virtual void OnInitCudaDevice(int deviceId) override;
};
