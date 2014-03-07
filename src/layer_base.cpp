#include "layer_base.h"

void LayerBase::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	if (!config)
		return;

	_name = config->Name;
	_learningRate = config->LearningRate;
	_momentum = config->Momentum;
	_weightDecay = config->WeightDecay;
}

LayerConfig::Ptr LayerBase::GetConfig() const
{
	auto ret = std::make_shared<LayerConfig>(_name);
	BuildConfig(*ret);
	return ret;
}

void LayerBase::BuildConfig(LayerConfig &config) const
{
	config.Name = _name;
	config.LearningRate = _learningRate;
	config.Momentum = _momentum;
	config.WeightDecay = _weightDecay;
}

void BindStruct(const axon::serialization::CStructBinder &binder, LayerBase &layer)
{
	binder("name", layer._name)
		("learnRate", layer._learningRate)
		("momentum", layer._momentum)
		("weightDecay", layer._weightDecay);
}