#include "layer_base.h"

void LayerBase::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	if (!config)
		return;

	_learningRate = config->LearningRate();
}

LayerConfig::Ptr LayerBase::GetConfig() const
{
	return std::make_shared<LayerConfig>(_name, _learningRate);
}

void BindStruct(const axon::serialization::CStructBinder &binder, LayerBase &layer)
{
	binder("name", layer._name)("learnRate", layer._learningRate);
}