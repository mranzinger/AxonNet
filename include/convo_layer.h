#pragma once

#include "linear_layer.h"



class ConvoLayer
	: public LayerBase
{
private:
	LinearLayer _linearLayer;
	size_t _windowSizeX, _windowSizeY;
	size_t _imageSizeX, _imageSizeY;
	size_t _strideX, _strideY;

public:
	enum PaddingMode
	{
		ZeroPad,
		NoPadding
	};

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
};