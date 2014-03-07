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

	size_t UpdateCt = 0;

	LinParams() { }
	LinParams(size_t numInputs, size_t numOutputs)
		: Weights(numOutputs, numInputs), Biases(numOutputs),
		  WeightsRunning(numOutputs, numInputs, 0), BiasRunning(numOutputs, 0),
		  WeightsIncrement(numOutputs, numInputs, 0), BiasIncrement(numOutputs, 0),
		  WeightDeltas(numOutputs, numInputs, 0), BiasDeltas(numOutputs, 0)
	{
		InitializeWeights(Weights, 0, 1);
		InitializeWeights(Biases, 0, 1);
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

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) override;
	virtual LayerConfig::Ptr GetConfig() const override;

	virtual void PrepareForThreads(size_t num) override;

	virtual void ApplyDeltas() override;
	virtual void ApplyDeltas(int threadIdx) override;

protected:
	void BuildConfig(LinearLayerConfig &config) const;

	LinParams &GetParams(int threadIdx);

private:
	void ApplyDeltas(LinParams &prms);
	void SyncToMaster(LinParams &prms);

};