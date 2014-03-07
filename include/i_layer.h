#pragma once

#include <memory>
#include <string>

#include <serialization\master.h>

#include "dll_include.h"
#include "math_util.h"

struct NEURAL_NET_API LayerConfig
{
public:
	typedef std::shared_ptr<LayerConfig> Ptr;

	std::string Name;
	Real LearningRate;
	Real Momentum;
	Real WeightDecay;

	virtual ~LayerConfig() { }

	LayerConfig() { }
	LayerConfig(std::string name)
		: Name(std::move(name)) { }

	friend void BindStruct(const axon::serialization::CStructBinder &binder, LayerConfig &config);
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
	virtual void SetMomentum(Real rate) = 0;
	virtual void SetWeightDecay(Real rate) = 0;
	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) = 0;
	virtual LayerConfig::Ptr GetConfig() const = 0;

	virtual void PrepareForThreads(size_t num) = 0;

	virtual void ApplyDeltas() = 0;
	virtual void ApplyDeltas(int threadIdx) = 0;
};



AXON_SERIALIZE_BASE_TYPE(ILayer)
AXON_SERIALIZE_BASE_TYPE(LayerConfig)