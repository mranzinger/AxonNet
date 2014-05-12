#pragma once

#include "linear_layer.h"

class NEURAL_NET_API ConvoLayerConfig
	: public LayerConfig
{
public:
	typedef std::shared_ptr<ConvoLayerConfig> Ptr;

	LayerConfig::Ptr LinearConfig;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, ConvoLayerConfig &config);
};

class NEURAL_NET_API ConvoLayer
	: public LayerBase
{
scope_private:
	LinearLayer _linearLayer;
	size_t _inputDepth;
	int _windowSizeX, _windowSizeY;
	int _padWidth, _padHeight;
	size_t _strideX, _strideY;

	std::vector<MultiParams> _threadWindows;

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

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

	virtual void ApplyDeltas() override;
	virtual void ApplyDeltas(int threadIdx) override;

	virtual void PrepareForThreads(size_t num) override;

	virtual void SyncWithHost() override;

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) override;
	virtual LayerConfig::Ptr GetConfig() const override;

	friend void ReadStruct(const axon::serialization::CStructReader &reader, ConvoLayer &layer);
	friend void WriteStruct(const axon::serialization::CStructWriter &binder, const ConvoLayer &layer);

scope_protected:
	void BuildConfig(ConvoLayerConfig &config) const;

scope_private:
	Params ComputePacked(int threadidx, const Params &input, bool isTraining);
	Params ComputePlanar(int threadIdx, const Params &input, bool isTraining);
};

