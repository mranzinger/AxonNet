/*
 * i_cu_weight_layer.cuh
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */


#pragma once

#include "weights.h"

class ICuWeightLayer
{
public:
	virtual ~ICuWeightLayer() { }

	virtual void ApplyGradient() = 0;

	virtual void SyncToDevice(const CWeights &hWeights);
	virtual void SyncToHost(CWeights &hWeights) const = 0;

	virtual void SetLearningRate(Real rate) = 0;
	virtual void SetMomentum(Real rate) = 0;
	virtual void SetWeightDecay(Real rate) = 0;
};


