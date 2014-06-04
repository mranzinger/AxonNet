/*
 * File description: cu_weights.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cu_weights.cuh"

CuWeights::CuWeights()
{
    SetDefaults();
}

CuWeights::CuWeights(cublasHandle_t handle, uint32_t numInputs, uint32_t numOutputs)
    : Weights(handle, numOutputs, numInputs),
      Biases(handle, numOutputs, 1)
{
    SetDefaults();
    RandInit();
}

CuWeights::CuWeights(CuMat weights, CuMat bias)
    : Weights(weights), Biases(bias)
{
    SetDefaults();
    Init();
}

void CuWeights::RandInit()
{
    CMatrix hWeights(Weights.Rows(), Weights.Cols());
    CMatrix hBiases(Weights.Rows(), 1);

    FanInitializeWeights(hWeights);
    FanInitializeWeights(hBiases);

    Weights.CopyToDevice(hWeights);
    Biases.CopyToDevice(hBiases);

    Init();
}

void CuWeights::Init()
{
    WeightsIncrement.ResizeLike(Weights);
    BiasIncrement.ResizeLike(Biases);

    WeightsGrad.ResizeLike(Weights);
    BiasGrad.ResizeLike(Biases);

    WeightsIncrement.SetConstant(0);
    BiasIncrement.SetConstant(0);
    WeightsGrad.SetConstant(0);
    BiasGrad.SetConstant(0);
}

void CuWeights::SetDefaults()
{
    LearningRate = 0.01f;
    Momentum = 0.9f;
    WeightDecay = 0.0005;
    DynamicLearningRate = 1.0f;
}

void CuWeights::ApplyGradient()
{
    if (Momentum)
    {
        WeightsIncrement *= Momentum;
        BiasIncrement *= Momentum;
    }
    else
    {
        WeightsIncrement.SetConstant(0);
        BiasIncrement.SetConstant(0);
    }

    if (WeightDecay)
    {
        Real wdFactor = WeightDecay * LearningRate;
        AddScaled(WeightsIncrement, 1.0f, Weights, -wdFactor, WeightsIncrement);
        AddScaled(BiasIncrement, 1.0f, Biases, -wdFactor, BiasIncrement);
    }

    Real learnRate = LearningRate * DynamicLearningRate;
    AddScaled(WeightsIncrement, 1.0f, WeightsGrad, -learnRate, WeightsIncrement);
    AddScaled(BiasIncrement, 1.0f, BiasGrad, -learnRate, BiasIncrement);

    Weights += WeightsIncrement;
    Biases += BiasIncrement;
}
