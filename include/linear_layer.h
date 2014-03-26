#pragma once

#include "layer_base.h"

class NEURAL_NET_API LinearLayerConfig
	: public LayerConfig
{
public:
	typedef std::shared_ptr<LinearLayerConfig> Ptr;

	Matrix Weights;
	Vector Biases;

	Matrix WeightsIncrement;
	Vector BiasesIncrement;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, LinearLayerConfig &config);
};

struct LinParams
{
	Matrix Weights;
	Vector Biases;

	Matrix WeightsRunning;
	Vector BiasRunning;

	Matrix WeightsIncrement;
	Vector BiasIncrement;

	Matrix WeightDeltas;
	Vector BiasDeltas;

	float LearningRate2 = 1;

	size_t UpdateCt = 0;

	LinParams() { }
	LinParams(size_t numInputs, size_t numOutputs)
		: Weights(numOutputs, numInputs), Biases(numOutputs),
		  WeightsRunning(numOutputs, numInputs), BiasRunning(numOutputs),
		  WeightsIncrement(numOutputs, numInputs), BiasIncrement(numOutputs),
		  WeightDeltas(numOutputs, numInputs), BiasDeltas(numOutputs)
	{
		//InitializeWeights(Weights, 0, 1);
		//InitializeWeights(Biases, 0, 1);
		FanInitializeWeights(Weights);
		FanInitializeWeights(Biases);

		WeightsRunning.setZero();
		BiasRunning.setZero();
		WeightsIncrement.setZero();
		BiasIncrement.setZero();
		WeightDeltas.setZero();
		BiasDeltas.setZero();
	}
};

typedef std::vector<LinParams> LinParamsList;

class NEURAL_NET_API LinearLayer
	: public LayerBase
{
protected:
	LinParams _master;
	LinParamsList _threadParams;

	size_t _updateInterval = 5;

public:
	typedef std::shared_ptr<LinearLayer> Ptr;

	LinearLayer() { }
	LinearLayer(std::string name, size_t numInputs, size_t numOutputs);
	LinearLayer(std::string name, Matrix weights, Vector biases);

	virtual std::string GetLayerType() const override {
		return "Linear Layer";
	}

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	void Compute(int threadIdx, const Params &input, Real *opBuff);
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
	MultiParams BackpropMany(int threadIdx, const MultiParams &lastInputs, const MultiParams &outputErrors);

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) override;
	virtual LayerConfig::Ptr GetConfig() const override;

	virtual void PrepareForThreads(size_t num) override;

	virtual void SyncWithHost() override;

	virtual void ApplyDeltas() override;
	virtual void ApplyDeltas(int threadIdx) override;

	size_t InputSize() const {
		return _master.Weights.innerSize();
	}
	size_t OutputSize() const {
		return _master.Weights.outerSize();
	}

	friend void BindStruct(const axon::serialization::CStructBinder &binder, LinearLayer &layer);

protected:
	void BuildConfig(LinearLayerConfig &config) const;

	LinParams &GetParams(int threadIdx);

private:
	void ApplyDeltas(LinParams &prms);
	void SyncToMaster(LinParams &prms);

};