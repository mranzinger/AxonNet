#pragma once

#include <memory>
#include <string>

#include <serialization\master.h>

#include "dll_include.h"
#include "math_util.h"

class NEURAL_NET_API LayerConfig
{
private:
	std::string _layerName;
	Real _learningRate;

public:
	typedef std::shared_ptr<LayerConfig> Ptr;

	virtual ~LayerConfig() { }

	LayerConfig() { }
	LayerConfig(std::string name)
		: _layerName(std::move(name)) { }
	LayerConfig(std::string name, Real learningRate)
		: _layerName(std::move(name)), _learningRate(learningRate) { }

	const std::string &Name() const {
		return _layerName;
	}

	Real LearningRate() const {
		return _learningRate;
	}
};

class NEURAL_NET_API ILayer
{
public:
	typedef std::shared_ptr<ILayer> Ptr;

	virtual ~ILayer();

	virtual const std::string &GetLayerName() const = 0;
	virtual std::string GetLayerType() const = 0;

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) = 0;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) = 0;

	virtual void SetLearningRate(Real rate) = 0;
	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) = 0;
	virtual LayerConfig::Ptr GetConfig() const = 0;
};



AXON_SERIALIZE_BASE_TYPE(ILayer)
AXON_SERIALIZE_BASE_TYPE(LayerConfig)