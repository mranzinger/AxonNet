/*
 * File description: matrix_math_tests.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

#include "math_util.h"
#include "test_helper.h"

using namespace std;
using namespace std::chrono;

TEST(MatrixMathTest, Multiply)
{
    RMatrix left = RMatrix::Random(20, 10);
    CMatrix right = CMatrix::Random(10, 12);

    auto start = system_clock::now();

    CMatrix mtResult;

    for (int i = 0; i < 10; ++i)
        mtResult = Gemm(left, right);

    auto end = system_clock::now();

    auto elapsedSecs = end - start;

    cout << "Gemm Time: " << elapsedSecs.count() << endl;

    start = system_clock::now();

    CMatrix eigResult;

    for (int i = 0; i < 10; ++i)
        eigResult = left * right;

    end = system_clock::now();

    elapsedSecs = end - start;

    cout << "Eigen Time: " << elapsedSecs.count() << endl;

    //AssertMatrixEquivalence(mtResult, eigResult);
}


