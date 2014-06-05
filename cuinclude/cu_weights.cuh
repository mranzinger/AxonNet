/*
 * File description: cu_weights.cuh
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "weights.h"
#include "cumat.cuh"

class CuWeights
{
public:
    CuMat Weights;
    CuMat Biases; // TODO: Implement a cuda vector type?

    CuMat WeightsIncrement;
    CuMat BiasIncrement;

    CuMat WeightsGrad;
    CuMat BiasGrad;

    Real LearningRate;
    Real Momentum;
    Real WeightDecay;

    Real DynamicLearningRate;

    CuWeights();
    CuWeights(CuContext handle, uint32_t numInputs, uint32_t numOutputs);
    CuWeights(CuMat weights, CuMat bias);

    CuWeights(CuContext handle, const CWeights &hWeights);

    CWeights ToHost() const;

    void CopyToDevice(const CWeights &hWeights);
    void CopyToHost(CWeights &hWeights) const;

    void RandInit();
    void Init();
    void SetDefaults();

    void ApplyGradient();
};


