#pragma once

#include "single_input_layer.h"

#include "cu_softmax_layer.cuh"

class NEURAL_NET_API SoftmaxLayer
	: public SingleInputLayer
{
scope_private:
	bool _checked;
	bool _costIsLogLoss;

	std::unique_ptr<CuSoftmaxLayer> _cuImpl;

scope_public:
	typedef std::shared_ptr<SoftmaxLayer> Ptr;

	SoftmaxLayer();
	explicit SoftmaxLayer(std::string name);
	SoftmaxLayer(std::string name, std::string inputName);

	virtual std::string GetLayerType() const override {
		return "Softmax Layer";
	}

	friend void BindStruct(const aser::CStructBinder &binder, SoftmaxLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
    virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

    virtual void OnInitCudaDevice(int deviceId) override;

scope_private:
	void EstablishContext();
};
