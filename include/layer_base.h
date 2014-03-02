#pragma once

#include "i_layer.h"

class NEURAL_NET_API LayerBase
	: public ILayer
{
protected:
	std::string _name;
	Real _learningRate;

public:
	typedef std::shared_ptr<LayerBase> Ptr;

	LayerBase() : _learningRate(1.0) { }
	LayerBase(std::string name) : _name(std::move(name)), _learningRate(1.0) { }

	virtual const std::string &GetLayerName() const override {
		return _name;
	}

	virtual void SetLearningRate(Real rate) override {
		_learningRate = rate;
	}

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config);
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, LayerBase &layer);
};

//namespace axon {
//	namespace serialization {

		//NEURAL_NET_API void BindStruct(const axon::serialization::CStructBinder &binder, LayerBase &layer);
//	}
//}