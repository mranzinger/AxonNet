/*
 * weight_layer.cpp
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */

#include "weight_layer.h"

#include "persist_util.h"

using namespace std;

WeightLayer::WeightLayer()
	: _gradConsumer(true)
{
}

WeightLayer::WeightLayer(size_t numInputs, size_t numOutputs)
	: _weights(numInputs, numOutputs), _gradConsumer(true)
{
}

WeightLayer::WeightLayer(CWeights weights, bool gradConsumer)
	: _weights(move(weights)), _gradConsumer(gradConsumer)
{
}

WeightLayer::WeightLayer(CMatrix weights, Vector bias, bool gradConsumer)
	: _weights(move(weights), move(bias)), _gradConsumer(gradConsumer)
{
}

void WeightLayer::SetLearningRate(Real rate)
{
    _weights.LearningRate = rate;

    if (_cuImpl)
    	_cuImpl->SetLearningRate(rate);
}

void WeightLayer::SetMomentum(Real rate)
{
    _weights.Momentum = rate;

    if (_cuImpl)
    	_cuImpl->SetMomentum(rate);
}

void WeightLayer::SetWeightDecay(Real rate)
{
    _weights.WeightDecay = rate;

    if (_cuImpl)
    	_cuImpl->SetWeightDecay(rate);
}

void WeightLayer::ApplyGradient()
{
    if (_gradConsumer)
    {
    	if (_cuImpl)
    		_cuImpl->ApplyGradient();
    	else
    		_weights.ApplyGradient();
    }
}

void WeightLayer::InitializeFromConfig(const LayerConfig::Ptr& config)
{
	auto win = dynamic_pointer_cast<WeightLayerConfig>(config);

	if (!win)
		throw runtime_error("The specified config is not for a weight layer.");

	_weights = CWeights(win->Weights, win->Biases);
	_weights.WeightsIncrement = win->WeightsIncrement;
	_weights.BiasIncrement = win->BiasesIncrement;

	if (_cuImpl)
		_cuImpl->SyncToDevice(_weights);
}

LayerConfig::Ptr WeightLayer::GetConfig() const
{
	auto ret = make_shared<WeightLayerConfig>();
	BuildConfig(*ret);
	return ret;
}

void WeightLayer::BuildConfig(WeightLayerConfig& config) const
{
	// If the cuda implementation is being used, then the weights
	// need to be synced back to the host before saving them
	if (_cuImpl)
		_cuImpl->SyncToHost(const_cast<CWeights&>(_weights));

	config.Weights = _weights.Weights;
	config.Biases = _weights.Biases;
	config.WeightsIncrement = _weights.WeightsIncrement;
	config.BiasesIncrement = _weights.BiasIncrement;
}

void WeightLayer::SetCudaImplementation(ICuWeightLayer* impl)
{
	_cuImpl = impl;

	_cuImpl->SyncToDevice(_weights);
	_cuImpl->SetLearningRate(_weights.LearningRate);
	_cuImpl->SetMomentum(_weights.Momentum);
	_cuImpl->SetWeightDecay(_weights.WeightDecay);
}

void WeightLayer::SyncToHost(bool gradToo)
{
    if (_cuImpl)
        _cuImpl->SyncToHost(_weights, gradToo);
}

void WeightLayer::SyncToDevice(bool gradToo)
{
    if (_cuImpl)
        _cuImpl->SyncToDevice(_weights, gradToo);
}

void BindStruct(const aser::CStructBinder &binder, WeightLayerConfig &config)
{
    BindStruct(binder, (LayerConfig&)config);

    binder("weights", config.Weights)
          ("biases", config.Biases)
          ("weightsIncrement", config.WeightsIncrement)
          ("biasIncrement", config.BiasesIncrement);
}

void WriteStruct(const aser::CStructWriter &writer, const WeightLayer &layer)
{
	writer("gradConsumer", layer._gradConsumer)
	      ("momentum", layer._weights.Momentum)
	      ("weightDecay", layer._weights.WeightDecay)
		  ("numInputs", layer._weights.Weights.cols())
		  ("numOutputs", layer._weights.Weights.rows());
}



void ReadStruct(const aser::CStructReader &reader, WeightLayer &layer)
{
	size_t numInputs, numOutputs;
	reader("gradConsumer", layer._gradConsumer)
          ("momentum", layer._weights.Momentum)
          ("weightDecay", layer._weights.WeightDecay)
		  ("numInputs", numInputs)
		  ("numOutputs", numOutputs);

	if (numInputs == 0 || numOutputs == 0)
		throw runtime_error("The dimensions of the weight layer must be specified.");

	layer._weights = CWeights(numInputs, numOutputs);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, WeightLayerConfig, WeightLayerConfig);


