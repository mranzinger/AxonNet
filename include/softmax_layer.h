#pragma once

#include "layer_base.h"

class NEURAL_NET_API SoftmaxLayer
	: public LayerBase
{
public:
	typedef std::shared_ptr<SoftmaxLayer> Ptr;

	SoftmaxLayer() { }
	SoftmaxLayer(std::string name) 
		: LayerBase(std::move(name)) { }

	virtual std::string GetLayerType() const override {
		return "Softmax Layer";
	}

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, SoftmaxLayer &layer);
};