/*
 * lin_params.cpp
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */

#include "weights.h"

CWeights::CWeights()
{
    SetDefaults();
}

CWeights::CWeights(size_t numInputs, size_t numOutputs)
		: Weights(numOutputs, numInputs), Biases(numOutputs)
{
    SetDefaults();
	RandInit();
}

CWeights::CWeights(CMatrix weights, Vector bias)
	: Weights(std::move(weights)), Biases(std::move(bias))
{
    SetDefaults();
	Init();
}

CWeights::CWeights(const CWeights& other)
	: Weights(other.Weights), Biases(other.Biases),
	  WeightsIncrement(other.WeightsIncrement),
	  BiasIncrement(other.BiasIncrement),
	  WeightsGrad(other.WeightsGrad),
	  BiasGrad(other.BiasGrad),
	  LearningRate(other.LearningRate),
	  Momentum(other.Momentum),
	  WeightDecay(other.WeightDecay),
	  DynamicLearningRate(other.DynamicLearningRate)
{
}

CWeights::CWeights(CWeights&& other)
{
	swap(*this, other);
}

CWeights& CWeights::operator =(CWeights other)
{
	swap(*this, other);
	return *this;
}

void swap(CWeights &a, CWeights &b)
{
	using std::swap;

	swap(a.Weights, b.Weights);
	swap(a.Biases, b.Biases);
	swap(a.WeightsIncrement, b.WeightsIncrement);
	swap(a.BiasIncrement, b.BiasIncrement);
	swap(a.WeightsGrad, b.WeightsGrad);
	swap(a.BiasGrad, b.BiasGrad);

	swap(a.LearningRate, b.LearningRate);
	swap(a.Momentum, b.Momentum);
	swap(a.WeightDecay, b.WeightDecay);
	swap(a.DynamicLearningRate, b.DynamicLearningRate);
}

void CWeights::RandInit()
{
	FanInitializeWeights(Weights);
	FanInitializeWeights(Biases);

	Init();
}

void CWeights::SetDefaults()
{
    LearningRate = 0.01f;
    Momentum = 0.9f;
    WeightDecay = 0.0005;
    DynamicLearningRate = 1.0f;
}

void CWeights::Init()
{
	WeightsIncrement.resizeLike(Weights);
	BiasIncrement.resizeLike(Biases);

	WeightsGrad.resizeLike(Weights);
	BiasGrad.resizeLike(Biases);

	WeightsIncrement.setZero();
	BiasIncrement.setZero();
	WeightsGrad.setZero();
	BiasGrad.setZero();
}

void CWeights::ApplyGradient()
{
	if (Momentum)
	{
		WeightsIncrement *= Momentum;
		BiasIncrement *= Momentum;
	}
	else
	{
		WeightsIncrement.setZero();
		BiasIncrement.setZero();
	}

	if (WeightDecay)
	{
		WeightsIncrement.noalias() -= (WeightDecay * LearningRate) * Weights;
		BiasIncrement.noalias() -= (WeightDecay * LearningRate) * Biases;
	}

	WeightsIncrement.noalias() -= (LearningRate * DynamicLearningRate) * WeightsGrad;
	BiasIncrement.noalias() -= (LearningRate * DynamicLearningRate) * BiasGrad;

	Weights.noalias() += WeightsIncrement;
	Biases.noalias() += BiasIncrement;
}


