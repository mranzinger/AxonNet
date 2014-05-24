
#include "linear_layer.h"
#include "fast_math.h"

#include "thread/parallel_for.h"

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
	int batchSize = input.BatchSize();
	int outputSize = _weights.Biases.size();
	int inputSize = input.size();

	Params ret(outputSize, 1, 1,
			   CMatrix(outputSize, batchSize));

	FastFor(GetThreadPool(), 0, batchSize, 1,
		[&, this] (int imageIdx)
	{
		UMapVector vecIpImage(const_cast<Real*>(input.Data.data()) + imageIdx * inputSize,
							  inputSize);
		UMapVector vecOpImage(ret.Data.data() + imageIdx * outputSize,
							  outputSize);

		vecOpImage.noalias() = _weights.Weights * vecIpImage + _weights.Biases;
	});

	return move(ret);
}

Params LinearLayer::SBackprop(const Params &lastInput, const Params &lastOutput,
							 const Params &outputErrors)
{
	int batchSize = lastInput.BatchSize();
	int outputSize = _weights.Biases.size();
	int inputSize = lastInput.size();

	RMatrix transWeights = _weights.Weights.transpose();

	Params inputErrors(lastInput,
					   CMatrix(lastInput.Data.rows(), lastInput.Data.cols()));

	FastFor(GetThreadPool(), 0, batchSize, 1,
			[&, this] (int imageIdx)
	{
		UMapVector vecOutputErrs(const_cast<Real*>(outputErrors.Data.data()) + imageIdx * outputSize,
								 outputSize);

		UMapVector vecInputErrs(inputErrors.Data.data() + imageIdx * inputSize, inputSize);

		// Calculate the input error
		vecInputErrs.noalias() = transWeights * vecOutputErrs;
	});

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


