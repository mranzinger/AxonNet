#pragma once

#include "i_layer.h"

class NEURAL_NET_API LayerBase
	: public virtual ILayer
{
scope_protected:
	std::string _name;
	Real _learningRate;
	Real _momentum;
	Real _weightDecay;

	NeuralNet *_net;

scope_public:
	typedef std::shared_ptr<LayerBase> Ptr;

	LayerBase() : _learningRate(1.0), _momentum(0.9), _weightDecay(0.0005), _net(nullptr) { }
	LayerBase(std::string name) : _name(std::move(name)), _learningRate(1.0), _momentum(0.9), _weightDecay(0.0005), _net(nullptr) { }

	virtual const std::string &GetLayerName() const override {
		return _name;
	}

	virtual void SetLearningRate(Real rate) override {
		_learningRate = rate;
	}
	virtual void SetMomentum(Real rate) override {
		_momentum = rate;
	}
	virtual void SetWeightDecay(Real rate) override {
		_weightDecay = rate;
	}

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config);
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, LayerBase &layer);

	virtual void ApplyGradient() override { }

	virtual void SetNet(NeuralNet *net) override { _net = net; }

#ifdef _UNIT_TESTS_
	void UTBackprop(ParamMap &computeMap, ParamMap &inputErrorMap)
	{
		Compute(computeMap, true);

		Backprop(computeMap, inputErrorMap);
	}
#endif

scope_protected:
	void BuildConfig(LayerConfig &config) const;

	Params *GetData(ParamMap &pMap, const std::string &name, bool enforce = true) const;
	const Params *GetData(const ParamMap &pMap, const std::string &name, bool enforce = true) const;
};
