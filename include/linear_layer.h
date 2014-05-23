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

	virtual void SetLearningRate(Real rate) override;
    virtual void SetMomentum(Real rate) override;
    virtual void SetWeightDecay(Real rate) override;

	virtual void ApplyGradient() override;

	friend void ReadStruct(const aser::CStructReader &reader, LinearLayer &layer);
    friend void WriteStruct(const aser::CStructWriter &binder, const LinearLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

};
