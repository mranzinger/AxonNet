/*
 * File description: binary_cost_tests.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include <random>

#include <gtest/gtest.h>

#include "logloss_cost.h"
#include "sum_sq_cost.h"

#include "test_helper.h"

using namespace std;

void TestCompute(SimpleCost &cost)
{
    Params input(new CMatrix(CMatrix::Random(10, 2)));
    Params labels(new CMatrix(CMatrix::Zero(10, 2)));

    mt19937 rnd(42);
    uniform_int_distribution<> dist(0, labels.Rows);
    for (uint32_t col = 0; col < labels.Cols; ++col)
        labels.GetHostMatrix()(dist(rnd), col) = 1.0f;

    CostMap hCost = cost.SCompute(input, labels);

    cost.SetDevicePreference(CudaDevicePreference::Create(0));

    CostMap dCost = cost.SCompute(input, labels);

    ASSERT_EQ(hCost.size(), dCost.size());

    auto hIter = hCost.begin();
    auto dIter = dCost.begin();

    for (auto hEnd = hCost.end(); hIter != hEnd; ++hIter, ++dIter)
    {
        ASSERT_EQ(hIter->first, dIter->first);

        ASSERT_FLOAT_EQ(hIter->second, dIter->second);
    }
}

void TestComputeGrad(SimpleCost &cost)
{
    Params input(new CMatrix(CMatrix::Random(10, 2)));
    Params labels(new CMatrix(CMatrix::Zero(10, 2)));

    mt19937 rnd(42);
    uniform_int_distribution<> dist(0, labels.Rows);
    for (uint32_t col = 0; col < labels.Cols; ++col)
        labels.GetHostMatrix()(dist(rnd), col) = 1.0f;

    Params hCost = cost.SComputeGrad(input, labels);

    cost.SetDevicePreference(CudaDevicePreference::Create(0));

    Params dCost = cost.SComputeGrad(input, labels);

    ASSERT_EQ(hCost.Rows, dCost.Rows);
    ASSERT_EQ(hCost.Cols, dCost.Cols);

    AssertMatrixEquivalence(hCost.GetHostMatrix(), dCost.GetHostMatrix());
}

TEST(LogLossCost, CudaCompute)
{
    LogLossCost cost;
    TestCompute(cost);
}

TEST(LogLossCost, CudaComputeGrad)
{
    LogLossCost cost;
    TestComputeGrad(cost);
}

TEST(SumSqCost, CudaCompute)
{
    SumSqCost cost;
    TestCompute(cost);
}

TEST(SumSqCost, CudaComputeGrad)
{
    SumSqCost cost;
    TestComputeGrad(cost);
}


