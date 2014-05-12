#include "linear_layer.h"
#include "persist_util.h"
#include "fast_math.h"

using namespace std;
using namespace axon::serialization;

LinearLayer::LinearLayer(string name, size_t numInputs, size_t numOutputs)
	: LayerBase(move(name)), _master(numInputs, numOutputs)
{

}

LinearLayer::LinearLayer(string name, RMatrix weights, Vector biases)
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
	//_master.WeightsIncrement = lin->WeightsIncrement;
	//_master.BiasIncrement = lin->BiasesIncrement;

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
	//config.WeightsIncrement = _master.WeightsIncrement;
	//config.BiasesIncrement = _master.BiasIncrement;
}

void LinearLayer::PrepareForThreads(size_t num)
{
	if (num > 1)
		_threadParams.resize(num, _master);
	else
		_threadParams.clear();
}

Params LinearLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	LinParams &prms = GetParams(threadIdx);

	Params ret(prms.Biases.size(), 1, 1, prms.Weights * input.Data);

	// The bias needs to be applied to each column
	ret.Data.colwise() += prms.Biases;

	return move(ret);
}

void LinearLayer::Compute(int threadIdx, const Params &input, Real *opBuff)
{
	LinParams &prms = GetParams(threadIdx);

	UMapVector op(opBuff, prms.Biases.size());

	op.noalias() = prms.Weights * input.Data;
	op.noalias() += prms.Biases;
}

Params LinearLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput,
							 const Params &outputErrors)
{
	LinParams &prms = GetParams(threadIdx);

	CMatrix inputErrors = prms.Weights.transpose() * outputErrors.Data;

	prms.WeightsGrad.noalias() = outputErrors.Data * lastInput.Data.transpose();
	prms.BiasGrad = outputErrors.Data.rowwise().sum();

	return move(inputErrors);
}

MultiParams LinearLayer::BackpropMany(int threadIdx, const MultiParams &lastInputs, const MultiParams &outputErrors)
{
	LinParams &prms = GetParams(threadIdx);

	MultiParams inputErrors(lastInputs.size());

	inputErrors[0].Data.noalias() = prms.Weights.transpose() * outputErrors[0].Data;

	prms.WeightsGrad.noalias() = outputErrors[0].Data * lastInputs[0].Data.transpose();
	prms.BiasGrad = outputErrors[0].Data;

	for (size_t i = 1; i < lastInputs.size(); ++i)
	{
		const Vector &lastInput = lastInputs[i].Data;
		const Vector &outputError = outputErrors[i].Data;

		inputErrors[i].Data.noalias() = prms.Weights.transpose() * outputError;

		prms.WeightsGrad.noalias() += outputError * lastInput.transpose();
		prms.BiasGrad += outputError;
	}

	prms.LearningRate2 = 1.0f / lastInputs.size();

	return move(inputErrors);
}

LinParams &LinearLayer::GetParams(int threadIdx)
{
	return _threadParams.empty() ? _master : _threadParams[threadIdx];
}

void LinearLayer::ApplyDeltas()
{
	if (!_threadParams.empty())
	{
		for (LinParams &params : _threadParams)
		{
			ApplyDeltas(params);
		}
	}
	else
	{
		ApplyDeltas(_master);
	}
}

void LinearLayer::ApplyDeltas(int threadIdx)
{
	ApplyDeltas(GetParams(threadIdx));
}

void LinearLayer::ApplyDeltas(LinParams &prms)
{
	/*prms.ExpWeightsGrad.noalias() = _decay * prms.ExpWeightsGrad +
									(1 - _decay) * (prms.WeightsGrad.cwiseProduct(prms.WeightsGrad));
	prms.ExpBiasGrad.noalias() = _decay * prms.ExpBiasGrad +
									(1 - _decay) * (prms.BiasGrad.cwiseProduct(prms.BiasGrad));

	RMatrix weightsInc = ((prms.ExpWeightsDelta.array() + _epsilon) /
						(prms.ExpWeightsGrad.array() + _epsilon))
							.sqrt().matrix()
							.cwiseProduct(prms.WeightsGrad);
	Vector biasInc = ((prms.ExpBiasDelta.array() + _epsilon) /
						(prms.ExpBiasGrad.array() + _epsilon))
							.sqrt().matrix()
							.cwiseProduct(prms.BiasGrad);

	prms.Weights.noalias() -= weightsInc;
	prms.Biases.noalias() -= biasInc;

	prms.ExpWeightsDelta.noalias() = _decay * prms.ExpWeightsDelta +
										(1 - _decay) * (weightsInc.cwiseProduct(weightsInc));
	prms.ExpBiasDelta.noalias() = _decay * prms.ExpBiasDelta +
										(1 - _decay) * (biasInc.cwiseProduct(biasInc));*/


	if (_momentum)
	{
		prms.WeightsIncrement *= _momentum;
		prms.BiasIncrement *= _momentum;
	}
	else
	{
		prms.WeightsIncrement.setZero();
		prms.BiasIncrement.setZero();
	}

	if (_weightDecay)
	{
		prms.WeightsIncrement.noalias() -= (_weightDecay * _learningRate) * prms.Weights;
		prms.BiasIncrement.noalias() -= (_weightDecay * _learningRate) * prms.Biases;
	}

	prms.WeightsIncrement.noalias() -= (_learningRate * prms.LearningRate2) * prms.WeightsGrad;
	prms.BiasIncrement.noalias() -= (_learningRate * prms.LearningRate2) * prms.BiasGrad;

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

	//prms.Weights.noalias() -= (_learningRate * prms.LearningRate2) * prms.WeightDeltas;
	//prms.Biases.noalias() -= (_learningRate * prms.LearningRate2) * prms.BiasGrad;
}

void LinearLayer::SyncWithHost()
{
	/*for (auto &prms : _threadParams)
	{
		prms.Weights = _master.Weights;
		prms.Biases = _master.Biases;

		prms.WeightsIncrement = _master.WeightsIncrement;
		prms.BiasIncrement = _master.BiasIncrement;
	}*/
}

void LinearLayer::SyncToMaster(LinParams &prms)
{
	if (_threadParams.empty())
		return;

	/*_master.Weights.noalias() += prms.WeightsRunning;
	_master.Biases.noalias() += prms.BiasRunning;

	prms.Weights = _master.Weights;
	prms.Biases = _master.Biases;

	prms.WeightsRunning.setZero();
	prms.BiasRunning.setZero();*/
}

void BindStruct(const CStructBinder &binder, LinearLayerConfig &config)
{
	BindStruct(binder, (LayerConfig&) config);

	binder("weights", config.Weights)
		("biases", config.Biases)
		("weightsInc", config.WeightsIncrement)
		("biasInc", config.BiasesIncrement);
}

void WriteStruct(const CStructWriter &writer, const LinearLayer &layer)
{
	WriteStruct(writer, (const LayerBase &)layer);

	writer
		("numInputs", layer._master.Weights.innerSize())
		("numOutputs", layer._master.Weights.outerSize());
}

void ReadStruct(const CStructReader &reader, LinearLayer &layer)
{
	ReadStruct(reader, (LayerBase &)layer);

	size_t numInputs = 0, numOutputs = 0;

	reader
		("numInputs", numInputs)
		("numOutputs", numOutputs);

	if (numInputs == 0 || numOutputs == 0)
		throw runtime_error("The dimensions of the linear layer must be specified.");

	layer._master = LinParams(numInputs, numOutputs);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, LinearLayerConfig, LinearLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, LinearLayer, LinearLayer);
