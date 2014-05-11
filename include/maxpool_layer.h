#pragma once

#include "layer_base.h"

class NEURAL_NET_API MaxPoolLayer
	: public LayerBase
{
private:
	size_t _windowSizeX, _windowSizeY;

public:
	MaxPoolLayer() = default;
	MaxPoolLayer(std::string name, size_t windowSizeX, size_t windowSizeY);

	virtual std::string GetLayerType() const override { return "Max Pool Layer"; }

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, MaxPoolLayer &layer);
};
