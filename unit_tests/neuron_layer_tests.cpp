/*
 * File description: neuron_layer_tests.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include <gtest/gtest.h>

#include "neuron_layer.h"

#include "test_helper.h"

using namespace std;

template<typename Fn>
void TestNeuronCompute()
{
    Params input(new CMatrix(CMatrix::Random(10, 10)));

    NeuronLayer<Fn> layer("");

    Params hostOutput = layer.SCompute(input, false);

    layer.SetDevicePreference(CudaDevicePreference::Create(0));

    Params cudaOutput = layer.SCompute(input, false);

    AssertMatrixEquivalence(hostOutput.GetHostMatrix(), cudaOutput.GetHostMatrix(),
                            0.001);
}

TEST(NeuronLayerTest, CudaCompute)
{
    TestNeuronCompute<LinearFn>();
    TestNeuronCompute<LogisticFn>();
    TestNeuronCompute<RectifierFn>();
    TestNeuronCompute<TanhFn>();
    TestNeuronCompute<RampFn>();
    TestNeuronCompute<SoftPlusFn>();
    TestNeuronCompute<HardTanhFn>();
}


