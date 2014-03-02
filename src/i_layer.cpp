#include "i_layer.h"

using namespace axon::serialization;

ILayer::~ILayer()
{

}

void BindStruct(const CStructBinder &binder, LayerConfig &config)
{
	binder("name", config._layerName)
		  ("learnRate", config._learningRate);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, LayerConfig, LayerConfig);