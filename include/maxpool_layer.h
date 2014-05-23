#pragma once

#include "single_input_layer.h"

class NEURAL_NET_API MaxPoolLayer
	: public SingleInputLayer
{
scope_private:
	size_t _windowSizeX, _windowSizeY;

scope_public:
	MaxPoolLayer() = default;
	MaxPoolLayer(std::string name, size_t windowSizeX, size_t windowSizeY);

	virtual std::string GetLayerType() const override { return "Max Pool Layer"; }

	friend void BindStruct(const aser::CStructBinder &binder, MaxPoolLayer &layer);

scope_protected:
    virtual Params SCompute(const Params &input, bool isTraining) override;
    virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

};
