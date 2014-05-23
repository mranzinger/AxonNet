#include "neuron_layer.h"

using namespace std;
using namespace axon::serialization;

template<typename Fn>
void BindStruct(const CStructBinder &binder, NeuronLayer<Fn> &layer)
{
	BindStruct(binder, (SingleInputLayer&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, LinearNeuronLayer, LinearNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, LogisticNeuronLayer, LogisticNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, RectifierNeuronLayer, RectifierNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, TanhNeuronLayer, TanhNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, RampNeuronLayer, RampNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, SoftPlusNeuronLayer, SoftPlusNeuronLayer);
AXON_SERIALIZE_DERIVED_TYPE(ILayer, HardTanhNeuronLayer, HardTanhNeuronLayer);
