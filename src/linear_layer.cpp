
#include "linear_layer.h"
#include "fast_math.h"

using namespace std;
using namespace axon::serialization;

LinearLayer::LinearLayer(string name, size_t numInputs, size_t numOutputs)
	: SingleInputLayer(move(name)), WeightLayer(numInputs, numOutputs)
{

}

LinearLayer::LinearLayer(string name, RMatrix weights, Vector biases)
	: SingleInputLayer(move(name)), WeightLayer(move(weights), move(biases))
{
}

void LinearLayer::InitializeFromConfig(const LayerConfig::Ptr& config)
{
	SingleInputLayer::InitializeFromConfig(config);
	WeightLayer::InitializeFromConfig(config);
}

LayerConfig::Ptr LinearLayer::GetConfig() const
{
	auto cfg = WeightLayer::GetConfig();

	SingleInputLayer::BuildConfig(*cfg);

	return cfg;
}

void LinearLayer::SetLearningRate(Real rate)
{
    WeightLayer::SetLearningRate(rate);
}

void LinearLayer::SetMomentum(Real rate)
{
    WeightLayer::SetMomentum(rate);
}

void LinearLayer::SetWeightDecay(Real rate)
{
    WeightLayer::SetWeightDecay(rate);
}

Params LinearLayer::SCompute(const Params &input, bool isTraining)
{
	Params ret(_weights.Biases.size(), 1, 1, _weights.Weights * input.Data);

	// The bias needs to be applied to each column
	ret.Data.colwise() += _weights.Biases;

	return move(ret);
}

Params LinearLayer::SBackprop(const Params &lastInput, const Params &lastOutput,
							 const Params &outputErrors)
{
	CMatrix inputErrors = _weights.Weights.transpose() * outputErrors.Data;

	_weights.WeightsGrad.noalias() = outputErrors.Data * lastInput.Data.transpose();
	_weights.BiasGrad = outputErrors.Data.rowwise().sum();

	return move(inputErrors);
}

void LinearLayer::ApplyGradient()
{
    SingleInputLayer::ApplyGradient();
    WeightLayer::ApplyGradient();
}

void WriteStruct(const aser::CStructWriter &writer, const LinearLayer &layer)
{
    WriteStruct(writer, (const SingleInputLayer &)layer);
    WriteStruct(writer, (const WeightLayer &)layer);
}
void ReadStruct(const aser::CStructReader &reader, LinearLayer &layer)
{
    ReadStruct(reader, (SingleInputLayer &)layer);
    ReadStruct(reader, (WeightLayer &)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, LinearLayer, LinearLayer);


