#pragma once

#include "single_input_layer.h"
#include "weight_layer.h"


class NEURAL_NET_API LinearLayer
	: public SingleInputLayer,
	  public WeightLayer
{
scope_public:
	typedef std::shared_ptr<LinearLayer> Ptr;

	LinearLayer() = default;
	LinearLayer(std::string name, size_t numInputs, size_t numOutputs);
	LinearLayer(std::string name, RMatrix weights, Vector biases);

	virtual std::string GetLayerType() const override {
		return "Linear Layer";
	}

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config);
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void BindStruct(const aser::CStructBinder &binder, LinearLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

};
