#include "fc_layer.h"
#include "persist_util.h"

using namespace axon::serialization;

void BindStruct(const CStructBinder &binder, FCLayerConfig &config)
{
	binder("weights", config.Weights)
		  ("biases", config.Biases);
}

template<typename Fn>
void BindStruct(const CStructBinder &binder, FCLayer<Fn> &layer)
{
	// Nothing really to serialize. It will all be captured in the config
}


AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, FCLayerConfig, FCLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, FCLayer<LinearFn>, LinearFCLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, FCLayer<LogisticFn>, LogisticFCLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, FCLayer<RectifierFn>, RectifierFCLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, FCLayer<TanhFn>, TanhFCLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, FCLayer<RampFn>, RampFCLayer);