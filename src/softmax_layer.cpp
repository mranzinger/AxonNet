#include "softmax_layer.h"

using namespace std;
using namespace axon::serialization;

Vector SoftmaxLayer::Compute(int threadIdx, const Vector &input, bool isTraining)
{

}

Vector SoftmaxLayer::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput,
							  const Vector &outputErrors)
{

}

void BindStruct(const CStructBinder &binder, SoftmaxLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);

	binder("numOutputs", layer._numOutputs);
}