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

CuWeights::CuWeights(CuContext handle, uint32_t numInputs, uint32_t numOutputs)
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

CuWeights::CuWeights(CuContext handle, const CWeights& hWeights)
    : Weights(handle, hWeights.Weights),
      Biases(handle, hWeights.Biases),
      WeightsIncrement(handle, hWeights.WeightsIncrement),
      BiasIncrement(handle, hWeights.BiasIncrement),
      LearningRate(hWeights.LearningRate),
      Momentum(hWeights.Momentum),
      WeightDecay(hWeights.WeightDecay),
      DynamicLearningRate(hWeights.DynamicLearningRate),
      WeightsGrad(handle),
      BiasGrad(handle)
{
    WeightsGrad.ResizeLike(Weights);
    BiasGrad.ResizeLike(Biases);
}

CWeights CuWeights::ToHost() const
{
    CWeights ret;
    CopyToHost(ret);

    ret.WeightsGrad.resizeLike(ret.Weights);
    ret.BiasGrad.resizeLike(ret.Biases);

    ret.LearningRate = LearningRate;
    ret.Momentum = Momentum;
    ret.WeightDecay = WeightDecay;
    ret.DynamicLearningRate = DynamicLearningRate;

    return ret;
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

void CuWeights::CopyToDevice(const CWeights& hWeights)
{
    Weights.CopyToDevice(hWeights.Weights);
    Biases.CopyToDevice(hWeights.Biases);

    WeightsIncrement.CopyToDevice(hWeights.WeightsIncrement);
    BiasIncrement.CopyToDevice(hWeights.BiasIncrement);

    WeightsGrad.ResizeLike(Weights);
    BiasGrad.ResizeLike(Biases);
}

void CuWeights::CopyToHost(CWeights& hWeights) const
{
    Weights.CopyToHost(hWeights.Weights);
    Biases.CopyToHost(hWeights.Biases);

    WeightsIncrement.CopyToHost(hWeights.WeightsIncrement);
    BiasIncrement.CopyToHost(hWeights.BiasIncrement);
}

void CuWeights::SetHandle(const CuContext& handle)
{
	Weights.SetHandle(handle);
	Biases.SetHandle(handle);

	WeightsGrad.SetHandle(handle);
	BiasGrad.SetHandle(handle);

	WeightsIncrement.SetHandle(handle);
	BiasIncrement.SetHandle(handle);
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
