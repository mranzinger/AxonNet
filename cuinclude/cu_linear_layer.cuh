/*
 * cu_linear_layer.cuh
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"
#include "weights.h"

#include "i_cu_weight_layer.cuh"

class CuLinearLayer
	: public virtual ICuWeightLayer
{
scope_public:
	CuLinearLayer(int deviceId);
	~CuLinearLayer();

	Params Compute(const Params &input) const;
	Params Backprop(const Params &lastInput, const Params &lastOutput,
				   const Params &outputErrors);

	virtual void ApplyGradient();

	virtual void SyncToDevice(const CWeights &hWeights);
	virtual void SyncToHost(CWeights &hWeights) const;

	virtual void SetLearningRate(Real rate);
	virtual void SetMomentum(Real rate);
	virtual void SetWeightDecay(Real rate);

scope_private:
	class Impl;
	Impl *_impl;
};


