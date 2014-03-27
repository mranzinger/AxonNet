#pragma once

#include "layer_base.h"

class NEURAL_NET_API SoftmaxLayer
	: public LayerBase
{
private:
	bool _checked;
	bool _costIsLogLoss;

public:
	typedef std::shared_ptr<SoftmaxLayer> Ptr;

	SoftmaxLayer() : SoftmaxLayer("") { }
	SoftmaxLayer(std::string name) 
		: LayerBase(std::move(name)), _checked(false), _costIsLogLoss(false) { }

	virtual std::string GetLayerType() const override {
		return "Softmax Layer";
	}

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, SoftmaxLayer &layer);

private:
	void EstablishContext();
};
