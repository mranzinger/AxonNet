#pragma once

#include <memory>
#include <string>

#include <serialization/master.h>

#include "params.h"

namespace aser = axon::serialization;

struct NEURAL_NET_API LayerConfig
{
public:
	typedef std::shared_ptr<LayerConfig> Ptr;

	std::string Name;

	virtual ~LayerConfig() { }

	LayerConfig() = default;
	LayerConfig(std::string name)
		: Name(std::move(name)) { }

	friend void BindStruct(const aser::CStructBinder &binder, LayerConfig &config);
};

class NeuralNet;

class NEURAL_NET_API ILayer
{
public:
	typedef std::shared_ptr<ILayer> Ptr;

	virtual ~ILayer();

	virtual const std::string &GetLayerName() const = 0;
	virtual std::string GetLayerType() const = 0;

	virtual void Compute(ParamMap &inputMap, bool isTraining) = 0;
	virtual void Backprop(const ParamMap &computeMap, ParamMap &inputErrorMap) = 0;

	virtual void SetLearningRate(Real rate) = 0;
	virtual void SetMomentum(Real rate) = 0;
	virtual void SetWeightDecay(Real rate) = 0;
	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) = 0;
	virtual LayerConfig::Ptr GetConfig() const = 0;

	virtual void ApplyGradient() = 0;

	virtual void SetNet(NeuralNet *net) = 0;
};

AXON_SERIALIZE_BASE_TYPE(ILayer)
AXON_SERIALIZE_BASE_TYPE(LayerConfig)
