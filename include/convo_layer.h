#pragma once

#include "linear_layer.h"

class NEURAL_NET_API ConvoLayer
	: public LayerBase
{
public:
	enum PaddingMode
	{
		ZeroPad,
		NoPadding
	};

private:
	LinearLayer _linearLayer;
	size_t _windowSizeX, _windowSizeY;
	size_t _strideX, _strideY;
	PaddingMode _padMode;

public:
	ConvoLayer() { }
	ConvoLayer(std::string name, 
				size_t inputDepth, size_t outputDepth, 
				size_t windowSizeX, size_t windowSizeY, 
				size_t strideX, size_t strideY, 
				PaddingMode padMode = NoPadding);

	virtual std::string GetLayerType() const override {
		return "Convo Layer";
	}

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;

	virtual void ApplyDeltas() override;
	virtual void ApplyDeltas(int threadIdx) override;

private:
	Params GetPaddedInput(const Params &input) const;
};