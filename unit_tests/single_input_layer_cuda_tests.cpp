/*
 * File description: simple_layer_cuda_tests.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include <iostream>

#include <gtest/gtest.h>

#include "softmax_layer.h"
#include "linear_layer.h"
#include "convo_layer.h"
#include "maxpool_layer.h"

#include "test_helper.h"

using namespace std;

void AssertParamsEq(const Params &a, const Params &b)
{
    ASSERT_EQ(a.Width, b.Width);
    ASSERT_EQ(a.Height, b.Height);
    ASSERT_EQ(a.Depth, b.Depth);
    ASSERT_EQ(a.Rows, b.Rows);
    ASSERT_EQ(a.Cols, b.Cols);

    AssertMatrixEquivalence(a.GetHostMatrix(), b.GetHostMatrix(), 0.001);
}

void TestCompute(SingleInputLayer &layer)
{
    Params input(32, 32, 5, new CMatrix(CMatrix::Random(32 * 32 * 5, 2)));

    Params hOutput = layer.SCompute(input, false);

    cout << "Host Matrix" << endl;
    cout << hOutput.GetHostMatrix().block(0, 0, 5, 2) << endl;

    layer.SetDevicePreference(CudaDevicePreference::Create(0));

    Params dOutput = layer.SCompute(input, false);

    AssertParamsEq(hOutput, dOutput);
}

template<typename TLayer, typename ...Params>
void TTestCompute(Params ...prms)
{
    TLayer layer(move(prms)...);
    TestCompute(layer);
}

/*TEST(SingleInputCuda, ComputeSoftmax)
{
    TTestCompute<SoftmaxLayer>("");
}

TEST(SingleInputCuda, ComputeLinearLayer)
{
    TTestCompute<LinearLayer>("", 32 * 32 * 5, 100);
}*/

TEST(SingleInputCuda, ComputeConvoLayer)
{
    TTestCompute<ConvoLayer>("",
                             5, 20,
                             7, 7,
                             2, 2,
                             3, 3);
}

TEST(SingleInputCuda, ComputeMaxPool)
{
    TTestCompute<MaxPoolLayer>("", 8, 8);
}

void TestBackprop(SingleInputLayer &layer)
{
    WeightLayer *wl = nullptr;
    CWeights initWeightsCopy;
    if (wl = dynamic_cast<WeightLayer*>(&layer))
    {
        initWeightsCopy = wl->_weights;
    }

    Params input(32, 32, 5, new CMatrix(CMatrix::Random(32 * 32 * 5, 2)));

    Params hOutput = layer.SCompute(input, true);

    Params outputErrors(hOutput, new CMatrix(CMatrix::Random(hOutput.Rows, hOutput.Cols)));

    Params hInputErrors = layer.SBackprop(input, hOutput, outputErrors);

    // Cache the updated weights, and reset them
    CWeights upWeights;
    if (wl)
    {
        upWeights = wl->_weights;
        wl->_weights = initWeightsCopy;
    }

    layer.SetDevicePreference(CudaDevicePreference::Create(0));

    Params dOutput = layer.SCompute(input, true);

    AssertParamsEq(hOutput, dOutput);

    Params dInputErrors = layer.SBackprop(input, dOutput, outputErrors);

    AssertParamsEq(hInputErrors, dInputErrors);

    if (wl)
    {
        wl->SyncToHost(true);

        AssertMatrixEquivalence(upWeights.Weights, wl->_weights.Weights, 0.001);
        AssertVectorEquivalence(upWeights.Biases, wl->_weights.Biases, 0.001);

        AssertMatrixEquivalence(upWeights.WeightsIncrement, wl->_weights.WeightsIncrement, 0.001);
        AssertVectorEquivalence(upWeights.BiasIncrement, wl->_weights.BiasIncrement, 0.001);

        AssertMatrixEquivalence(upWeights.WeightsGrad, wl->_weights.WeightsGrad, 0.001);
        AssertVectorEquivalence(upWeights.BiasGrad, wl->_weights.BiasGrad, 0.001);
    }
}

template<typename TLayer, typename ...Params>
void TTestBackprop(Params ...prms)
{
    TLayer layer(move(prms)...);
    TestBackprop(layer);
}

/*TEST(SingleInputCuda, BackpropSoftmax)
{
    TTestBackprop<SoftmaxLayer>("");
}*/

TEST(SingleInputCuda, BackpropLinear)
{
    TTestBackprop<LinearLayer>("", 32 * 32 * 5, 100);
}

TEST(SingleInputCuda, BackpropConvo)
{
    TTestBackprop<ConvoLayer>("", 5, 20, 4, 4, 1, 1);
}

TEST(SingleInputCuda, BackpropMaxPool)
{
    TTestBackprop<MaxPoolLayer>("", 8, 8);
}
