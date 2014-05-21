#pragma once

#include "single_input_layer.h"
#include "weight_layer.h"

class NEURAL_NET_API ConvoLayer
	: public SingleInputLayer,
	  public WeightLayer
{
scope_private:
	size_t _inputDepth;
	int _windowSizeX, _windowSizeY;
	int _padWidth, _padHeight;
	size_t _strideX, _strideY;

scope_public:
	ConvoLayer() = default;
	ConvoLayer(std::string name, 
				size_t inputDepth, size_t outputDepth, 
				size_t windowSizeX, size_t windowSizeY, 
				size_t strideX, size_t strideY, 
				int padWidth = 0, int padHeight = 0);
	ConvoLayer(std::string name,
				RMatrix linWeights, Vector linBias,
				size_t windowSizeX, size_t windowSizeY,
				size_t strideX, size_t strideY,
				int padWidth = 0, int padHeight = 0);

	virtual std::string GetLayerType() const override {
		return "Convo Layer";
	}

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config);
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void ReadStruct(const aser::CStructReader &reader, ConvoLayer &layer);
	friend void WriteStruct(const aser::CStructWriter &binder, const ConvoLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

scope_private:
	Params ComputePacked(const Params &input, bool isTraining);
	Params ComputePlanar(const Params &input, bool isTraining);
};

