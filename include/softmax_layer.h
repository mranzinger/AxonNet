#pragma once

#include "layer_base.h"

class NEURAL_NET_API SoftmaxLayer
	: public LayerBase
{
private:
	size_t _numOutputs;

public:
	typedef std::shared_ptr<SoftmaxLayer> Ptr;

	SoftmaxLayer()
		: _numOutputs(std::numeric_limits<size_t>::max()) { }
	SoftmaxLayer(std::string name) 
		: LayerBase(std::move(name)), _numOutputs(std::numeric_limits<size_t>::max()) { }
	SoftmaxLayer(std::string name, size_t numOutputs) 
		: LayerBase(std::move(name)), _numOutputs(numOutputs) { }

	virtual std::string GetLayerType() const override {
		return "Softmax Layer";
	}

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, SoftmaxLayer &layer);
};