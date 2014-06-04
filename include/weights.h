/*
 * lin_params.h
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */

#pragma once

#include "math_util.h"

class CWeights
{
public:
	CMatrix Weights;
	Vector Biases;

	CMatrix WeightsIncrement;
	Vector BiasIncrement;

	CMatrix WeightsGrad;
	Vector BiasGrad;

	Real LearningRate;
	Real Momentum;
	Real WeightDecay;

	Real DynamicLearningRate;

	CWeights();
	CWeights(size_t numInputs, size_t numOutputs);
	CWeights(CMatrix weights, Vector bias);
	CWeights(const CWeights &other);

#ifndef _CUDA_COMPILE_
	CWeights(CWeights &&other);
#endif

	CWeights &operator=(CWeights other);

	friend void swap(CWeights &a, CWeights &b);

	void RandInit();
	void Init();
	void SetDefaults();

	void ApplyGradient();
};


