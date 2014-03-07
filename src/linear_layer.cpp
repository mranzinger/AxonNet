#include "linear_layer.h"
#include "persist_util.h"

using namespace std;
using namespace axon::serialization;

LinearLayer::LinearLayer(string name, size_t numInputs, size_t numOutputs)
	: LayerBase(move(name)), _master(numInputs, numOutputs)
{

}

LinearLayer::LinearLayer(string name, Matrix weights, Vector biases)
	: LayerBase(move(name))
{
	_master.Weights.swap(weights);
	_master.Biases.swap(biases);
}

void LinearLayer::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	LayerBase::InitializeFromConfig(config);

	auto lin = dynamic_pointer_cast<LinearLayerConfig>(config);

	if (!lin)
		throw runtime_error("The specified config is not for a linear layer.");

	_master.Weights = lin->Weights;
	_master.Biases = lin->Biases;
	_master.WeightsIncrement = lin->WeightsIncrement;
	_master.BiasIncrement = lin->BiasesIncrement;

	for (auto &thrd : _threadParams)
		thrd = _master;
}

LayerConfig::Ptr LinearLayer::GetConfig() const
{
	auto ret = make_shared<LinearLayerConfig>();
	BuildConfig(*ret);
	return ret;
}

void LinearLayer::BuildConfig(LinearLayerConfig &config) const
{
	LayerBase::BuildConfig(config);

	config.Weights = _master.Weights;
	config.Biases = _master.Biases;
	config.WeightsIncrement = _master.WeightsIncrement;
	config.BiasesIncrement = _master.BiasIncrement;
}

void LinearLayer::PrepareForThreads(size_t num)
{
	_threadParams.resize(num, _master);
}

Vector LinearLayer::Compute(int threadIdx, const Vector &input, bool isTraining)
{
	LinParams &prms = GetParams(threadIdx);

	return prms.Weights * input + prms.Biases;
}

Vector LinearLayer::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput,
							 const Vector &outputErrors)
{
	LinParams &prms = GetParams(threadIdx);

	Vector inputErrors = prms.Weights.transpose() * outputErrors;

	prms.WeightDeltas = outputErrors * lastInput.transpose();
	prms.BiasDeltas = outputErrors;

	return move(inputErrors);
}

LinParams &LinearLayer::GetParams(int threadIdx)
{
	return _threadParams.empty() ? _master : _threadParams[threadIdx];
}

void LinearLayer::ApplyDeltas()
{
	for (LinParams &params : _threadParams)
	{
		ApplyDeltas(params);
	}
}

void LinearLayer::ApplyDeltas(int threadIdx)
{
	ApplyDeltas(_threadParams[threadIdx]);
}

void LinearLayer::ApplyDeltas(LinParams &prms)
{
	if (_momentum)
	{
		prms.WeightsIncrement.noalias() += _momentum * prms.WeightsIncrement;
		prms.BiasIncrement.noalias() += _momentum * prms.BiasIncrement;
	}

	if (_weightDecay)
	{
		// TODO
	}

	prms.WeightsIncrement.noalias() -= _learningRate * prms.WeightDeltas;
	prms.BiasIncrement.noalias() -= _learningRate * prms.BiasDeltas;

	if (!_threadParams.empty())
	{
		prms.WeightsRunning.noalias() += prms.WeightsIncrement;
		prms.BiasRunning.noalias() += prms.BiasIncrement;
	}

	prms.Weights.noalias() += prms.WeightsIncrement;
	prms.Biases.noalias() += prms.BiasIncrement;

	if (++prms.UpdateCt == _updateInterval)
	{
		prms.UpdateCt = 0;
		SyncToMaster(prms);
	}
}

void LinearLayer::SyncToMaster(LinParams &prms)
{
	if (_threadParams.empty())
		return;

	_master.Weights.noalias() += prms.WeightsRunning;
	_master.Biases.noalias() += prms.BiasRunning;

	prms.Weights = _master.Weights;
	prms.Biases = _master.Biases;

	prms.WeightsRunning.setZero();
	prms.BiasRunning.setZero();
}

void BindStruct(const CStructBinder &binder, LinearLayerConfig &config)
{
	BindStruct(binder, (LayerConfig&) config);

	binder("weights", config.Weights)
		("biases", config.Biases)
		("weightsInc", config.WeightsIncrement)
		("biasInc", config.BiasesIncrement);
}

void BindStruct(const CStructBinder &binder, LinearLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, LinearLayerConfig, LinearLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, LinearLayer, LinearLayer);
