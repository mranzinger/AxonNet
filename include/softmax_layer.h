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

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, SoftmaxLayer &layer);
};