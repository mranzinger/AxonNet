#include "i_layer.h"

using namespace axon::serialization;

ILayer::~ILayer()
{

}

void BindStruct(const CStructBinder &binder, LayerConfig &config)
{
	binder("name", config.Name)
		  ("learnRate", config.LearningRate)
		  ("momentum", config.Momentum)
		  ("weightDecay", config.WeightDecay);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, LayerConfig, LayerConfig);