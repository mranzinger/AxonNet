/*
 * cu_convo_layer.cuh
 *
 *  Created on: Jun 15, 2014
 *      Author: mike
 */

#pragma once

#include "params.h"
#include "weights.h"

#include "i_cu_weight_layer.h"

class CuConvoLayer
	: public ICuWeightLayer
{
scope_public:
	CuConvoLayer(int deviceId);
	~CuConvoLayer();

	Params Compute(const Params &input) const;
	Params Backprop(const Params &lastInput, const Params &lastOutput,
				   const Params &outputErrors);

	virtual void ApplyGradient();

	virtual void SyncToDevice(const CWeights &hWeights, bool gradToo = false);
	virtual void SyncToHost(CWeights &hWeights, bool gradToo = false) const;

	virtual void SetLearningRate(Real rate);
	virtual void SetMomentum(Real rate);
	virtual void SetWeightDecay(Real rate);

scope_private:
	class Impl;
	Impl *_impl;
};


