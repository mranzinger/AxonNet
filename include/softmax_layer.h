#pragma once

#include "single_input_layer.h"

class NEURAL_NET_API SoftmaxLayer
	: public SingleInputLayer
{
private:
	bool _checked;
	bool _costIsLogLoss;

public:
	typedef std::shared_ptr<SoftmaxLayer> Ptr;

	SoftmaxLayer();
	SoftmaxLayer(std::string name);
	SoftmaxLayer(std::string name, std::string inputName);

	virtual std::string GetLayerType() const override {
		return "Softmax Layer";
	}

	friend void BindStruct(const aser::CStructBinder &binder, SoftmaxLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
    virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

private:
	void EstablishContext();
};
