
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
	if (_cuImpl)
		return _cuImpl->Compute(input);

	const uint32_t batchSize = input.Cols;
	const uint32_t outputSize = _weights.Biases.size();
	const uint32_t inputSize = input.Rows;

	Params ret(outputSize, 1, 1,
			   new CMatrix(outputSize, batchSize));

	const CMatrix &mInput = input.GetHostMatrix();
	CMatrix &mOutput = ret.GetHostMatrix();

	FastFor(GetThreadPool(), 0u, batchSize, 1u,
		[&, this] (int imageIdx)
	{
		UMapVector vecIpImage(const_cast<Real*>(mInput.data()) + imageIdx * inputSize,
							  inputSize);
		UMapVector vecOpImage(mOutput.data() + imageIdx * outputSize,
							  outputSize);

		vecOpImage.noalias() = _weights.Weights * vecIpImage + _weights.Biases;
	});

	return move(ret);
}

Params LinearLayer::SBackprop(const Params &lastInput, const Params &lastOutput,
							 const Params &outputErrors)
{
	if (_cuImpl)
		return _cuImpl->Backprop(lastInput, lastOutput, outputErrors);

	const uint32_t batchSize = lastInput.Cols;
	const uint32_t outputSize = _weights.Biases.size();
	const uint32_t inputSize = lastInput.Rows;

	CMatrix transWeights = _weights.Weights.transpose();

	Params inputErrors(lastInput,
					   new CMatrix(lastInput.Rows, lastInput.Cols));

	const CMatrix &mInput = lastInput.GetHostMatrix();
	const CMatrix &mOutputErrors = outputErrors.GetHostMatrix();

	CMatrix &mInputErrors = inputErrors.GetHostMatrix();

	FastFor(GetThreadPool(), 0u, batchSize, 1u,
			[&, this] (int imageIdx)
	{
		UMapVector vecOutputErrs(const_cast<Real*>(mOutputErrors.data()) + imageIdx * outputSize,
								 outputSize);

		UMapVector vecInputErrs(mInputErrors.data() + imageIdx * inputSize, inputSize);

		// Calculate the input error
		vecInputErrs.noalias() = transWeights * vecOutputErrs;
	});

	_weights.WeightsGrad.noalias() = mOutputErrors * mInput.transpose();
	_weights.BiasGrad = mOutputErrors.rowwise().sum();

	return move(inputErrors);
}

void LinearLayer::ApplyGradient()
{
    SingleInputLayer::ApplyGradient();
    WeightLayer::ApplyGradient();
}

void LinearLayer::OnInitCudaDevice(int deviceId)
{
	_cuImpl.reset(new CuLinearLayer(deviceId));

	// Inform the weight layer that a cuda implementation is present
	SetCudaImplementation(_cuImpl.get());
}

void WriteStruct(const aser::CStructWriter &writer, const LinearLayer &layer)
{
	WriteStruct(writer, (const WeightLayer &)layer);
    WriteStruct(writer, (const SingleInputLayer &)layer);
}
void ReadStruct(const aser::CStructReader &reader, LinearLayer &layer)
{
	ReadStruct(reader, (WeightLayer &)layer);
    ReadStruct(reader, (SingleInputLayer &)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, LinearLayer, LinearLayer);


