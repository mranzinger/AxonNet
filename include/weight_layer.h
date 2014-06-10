/*
 * weight_layer.h
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */

#pragma once

#include "i_layer.h"

#include "weights.h"

#include "i_cu_weight_layer.h"

class NEURAL_NET_API WeightLayerConfig
	: public LayerConfig
{
public:
	typedef std::shared_ptr<LayerConfig> Ptr;

	RMatrix Weights;
	Vector Biases;

	RMatrix WeightsIncrement;
	Vector BiasesIncrement;

	friend void BindStruct(const aser::CStructBinder &binder,
						   WeightLayerConfig &config);
};

class NEURAL_NET_API WeightLayer
	: public virtual ILayer
{
scope_protected:
	CWeights _weights;
	bool _gradConsumer;

scope_public:
	typedef std::shared_ptr<WeightLayer> Ptr;

	WeightLayer();
	WeightLayer(size_t numInputs, size_t numOutputs);
	WeightLayer(CWeights weights, bool gradConsumer = true);
	WeightLayer(RMatrix weights, Vector bias, bool gradConsumer = true);

	size_t InputSize() const {
		return _weights.Weights.cols();
	}
	size_t OutputSize() const {
		return _weights.Weights.rows();
	}

	bool GradConsumer() const { return _gradConsumer; }
	void SetGradConsumer(bool val) { _gradConsumer = val; }

	virtual void SetLearningRate(Real rate) override;
	virtual void SetMomentum(Real rate) override;
	virtual void SetWeightDecay(Real rate) override;

	virtual void ApplyGradient() override;

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) override;
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void WriteStruct(const aser::CStructWriter &writer, const WeightLayer &layer);
	friend void ReadStruct(const aser::CStructReader &reader, WeightLayer &layer);

scope_protected:
	void BuildConfig(WeightLayerConfig &config) const;

	void SetCudaImplementation(ICuWeightLayer *impl);

scope_private:
	ICuWeightLayer *_cuImpl = nullptr;
};


