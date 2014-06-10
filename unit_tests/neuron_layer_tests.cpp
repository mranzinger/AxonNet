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

template<typename Fn>
void TestNeuronDerivative()
{
    Params input(new CMatrix(CMatrix::Random(10, 10)));
    Params outputErrors(new CMatrix(CMatrix::Random(10, 10)));

    NeuronLayer<Fn> layer("");

    Params hOutput = layer.SCompute(input, true);

    Params hInputErrors = layer.SBackprop(input, hOutput, outputErrors);

    layer.SetDevicePreference(CudaDevicePreference::Create(0));

    Params dOutput = layer.SCompute(input, true);

    Params dInputErrors = layer.SBackprop(input, dOutput, outputErrors);

    AssertMatrixEquivalence(hOutput.GetHostMatrix(), dOutput.GetHostMatrix(), 0.001);
    AssertMatrixEquivalence(hInputErrors.GetHostMatrix(), dInputErrors.GetHostMatrix(), 0.001);
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

TEST(NeuronLayerTest, CudaBackprop)
{
    TestNeuronDerivative<LinearFn>();
    TestNeuronDerivative<LogisticFn>();
    TestNeuronDerivative<RectifierFn>();
    TestNeuronDerivative<TanhFn>();
    TestNeuronDerivative<RampFn>();
    TestNeuronDerivative<SoftPlusFn>();
    TestNeuronDerivative<HardTanhFn>();
}
