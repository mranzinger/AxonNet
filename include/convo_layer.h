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
public:
	enum PaddingMode
	{
		ZeroPad,
		NoPadding,
		ConstantPad
	};

private:
	LinearLayer _linearLayer;
	size_t _inputDepth;
	size_t _windowSizeX, _windowSizeY;
	size_t _strideX, _strideY;
	Vector _constPad;
	PaddingMode _padMode;

	std::vector<MultiParams> _threadWindows;

public:
	ConvoLayer() = default;
	ConvoLayer(std::string name, 
				size_t inputDepth, size_t outputDepth, 
				size_t windowSizeX, size_t windowSizeY, 
				size_t strideX, size_t strideY, 
				PaddingMode padMode = NoPadding,
				Vector constPad = Vector());
	ConvoLayer(std::string name,
				Matrix linWeights, Vector linBias,
				size_t windowSizeX, size_t windowSizeY,
				size_t strideX, size_t strideY,
				PaddingMode padMode = NoPadding,
				Vector constPad = Vector());

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

	void SetConstantPad(Vector pad);

	friend void BindStruct(const axon::serialization::CStructBinder &binder, ConvoLayer &layer);

protected:
	void BuildConfig(ConvoLayerConfig &config) const;

private:
	Params ComputePacked(int threadidx, const Params &input, bool isTraining);
	Params ComputePlanar(int threadIdx, const Params &input, bool isTraining);

	Params GetPaddedInput(const Params &input) const;
	Params GetZeroPaddedInput(const Params &reference) const;
};
