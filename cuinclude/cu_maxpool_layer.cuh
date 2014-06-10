/*
 * cu_maxpool_layer.cuh
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"

class CuMaxPoolLayer
{
scope_private:
	CuContext _handle;

	uint32_t _windowSizeX, _windowSizeY;

	uint32_t _stepX, _stepY;

scope_public:
	CuMaxPoolLayer(int deviceId);
	CuMaxPoolLayer(int deviceId, uint32_t windowSizeX, uint32_t windowSizeY);
	CuMaxPoolLayer(int deviceId, uint32_t windowSizeX, uint32_t windowSizeY,
				   uint32_t stepX, uint32_t stepY);

	Params Compute(const Params &input);
	Params Backprop(const Params &input, const Params &lastOutput,
					const Params &outputErrors);

	void SetWindowSize(uint32_t windowSizeX, uint32_t windowSizeY);
	void SetStepSize(uint32_t stepX, uint32_t stepY);
	void ResetStepSize();
	void InitDevice(int deviceId);

scope_private:
	void EnsureStep();
};


