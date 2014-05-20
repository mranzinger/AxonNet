/*
 * File description: matrix_math.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <immintrin.h>
#include <assert.h>

#include "math_util.h"



template<bool Add>
inline void Gemm(const RMatrix &leftMat, const CMatrix &rightMat, CMatrix &opMat)
{
    using namespace std;

    assert(leftMat.cols() == rightMat.rows());

    size_t numRows = leftMat.rows(),
           numCols = rightMat.cols();

    size_t leftCols = leftMat.cols(),
           rightRows = rightMat.rows();

    opMat.resize(numRows, numCols);

    // Get the number of elements that can be processed
    // with sse vectors
    size_t numSSE = rightRows & ~0x07;
    size_t numLeftover = rightRows & 0x07;

    Real *opVal = opMat.data();

    // Start with a naive approach
    for (size_t col = 0; col < numCols; ++col)
    {
        const Real *const sRightCol = rightMat.data() + col * rightRows;
        const Real *const sRightEndCol = sRightCol + rightRows;

        for (size_t row = 0; row < numRows; ++row, ++opVal)
        {
            const Real *const sLeftRow = leftMat.data() + row * leftCols;
            const Real *const sLeftEndRow = sLeftRow + leftCols;

            const Real *sCurrRightCol = sRightCol,
                       *sCurrLeftRow = sLeftRow;
            const Real *sCurrSSEEnd = sCurrRightCol + numSSE;
            const Real *sCurrEnd = sCurrRightCol + rightRows;

            // Use 2 different sums to eliminate the data dependency per iteration
            __m128 vSum1 = _mm_setzero_ps(),
                   vSum2 = _mm_setzero_ps();

            for (; sCurrRightCol != sCurrSSEEnd; sCurrRightCol += 8, sCurrLeftRow += 8)
            {
                const __m128 vCol1 = _mm_loadu_ps(sCurrRightCol);
                const __m128 vCol2 = _mm_loadu_ps(sCurrRightCol + 4);
                const __m128 vRow1 = _mm_loadu_ps(sCurrLeftRow);
                const __m128 vRow2 = _mm_loadu_ps(sCurrLeftRow + 4);

                vSum1 = _mm_add_ps(vSum1, _mm_mul_ps(vCol1, vRow1));
                vSum2 = _mm_add_ps(vSum2, _mm_mul_ps(vCol2, vRow2));
            }

            union
            {
                __m128 vHsum;
                Real sHsum[4];
            };

            // Combine the sums
            vHsum = _mm_add_ps(vSum1, vSum2);

            // The result of the first hadd is (v2_2 + v2_3, v2_0 + v2_1, v1_2 + v1_3, v1_0 + v1_1)
            vHsum = _mm_hadd_ps(vHsum, vHsum);
            vHsum = _mm_hadd_ps(vHsum, vHsum);

            Real sSum = sHsum[0];

            // Use Duff's device to finish off the stragglers
            switch (numLeftover)
            {
            case 7:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 6:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 5:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 4:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 3:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 2:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            case 1:
                sSum += *sCurrLeftRow++ * *sCurrRightCol++;
            }

            if (Add)
                *opVal += sSum;
            else
                *opVal = sSum;
        }
    }
}

inline CMatrix Gemm(const RMatrix &leftMat, const CMatrix &rightMat)
{
    CMatrix ret;
    Gemm<false>(leftMat, rightMat, ret);
    return ret;
}

