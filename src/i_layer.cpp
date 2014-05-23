#include "i_layer.h"

using namespace axon::serialization;

ILayer::~ILayer()
{

}

void BindStruct(const CStructBinder &binder, LayerConfig &config)
{
	binder("name", config.Name);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, LayerConfig, LayerConfig);
