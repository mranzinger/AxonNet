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
	RMatrix Weights;
	Vector Biases;

	RMatrix WeightsIncrement;
	Vector BiasIncrement;

	RMatrix WeightsGrad;
	Vector BiasGrad;

	Real LearningRate = 0.01f;
	Real Momentum = 0.9;
	Real WeightDecay = 0.0005;

	Real DynamicLearningRate = 1;

	CWeights() = default;
	CWeights(size_t numInputs, size_t numOutputs);
	CWeights(RMatrix weights, Vector bias);
	CWeights(const CWeights &other);
	CWeights(CWeights &&other);

	CWeights &operator=(CWeights other);

	friend void swap(CWeights &a, CWeights &b);

	void RandInit();
	void Init();

	void ApplyGradient();
};


