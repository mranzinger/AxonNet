#pragma once

#include "i_layer.h"

class NEURAL_NET_API LayerBase
	: public ILayer
{
protected:
	std::string _name;
	Real _learningRate;
	Real _momentum;
	Real _weightDecay;

public:
	typedef std::shared_ptr<LayerBase> Ptr;

	LayerBase() : _learningRate(1.0), _momentum(0), _weightDecay(0) { }
	LayerBase(std::string name) : _name(std::move(name)), _learningRate(1.0), _momentum(0), _weightDecay(0) { }

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

	virtual void PrepareForThreads(size_t num) override { }

	virtual void ApplyDeltas() override { }
	virtual void ApplyDeltas(int threadIdx) override { }

protected:
	void BuildConfig(LayerConfig &config) const;
};