#include "maxpool_layer.h"

using namespace std;
using namespace axon::serialization;

using Eigen::MatrixXi;

MaxPoolLayer::MaxPoolLayer(string name, size_t windowSizeX, size_t windowSizeY)
	: LayerBase(move(name)), _windowSizeX(windowSizeX), _windowSizeY(windowSizeY)
{
	PrepareForThreads(1);
}

Params MaxPoolLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{

}

Params MaxPoolLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{

}

void MaxPoolLayer::PrepareForThreads(size_t num)
{
	_threadIndexes.resize(num);
}

void BindStruct(const CStructBinder &binder, MaxPoolLayer &layer)
{
	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MaxPoolLayer, MaxPoolLayer);