/*
 * cu_linear_layer.cuh
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"
#include "weights.h"

class CuLinearLayer
{
scope_public:
	CuLinearLayer(int deviceId);
	~CuLinearLayer();

	Params Compute(const Params &input) const;
	Params Backprop(const Params &lastInput, const Params &lastOutput,
				   const Params &outputErrors);

	void ApplyGradient();

	void SyncToDevice(const CWeights &hWeights);
	void SyncToHost(CWeights &hWeights) const;

scope_private:
	class Impl;
	Impl *_impl;
};


