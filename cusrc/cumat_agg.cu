/*
 * File description: cumat_agg.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat.cuh"

CuRowwiseOperator::CuRowwiseOperator(const CuMat &mat)
    : Mat(mat) { }

CuColwiseOperator::CuColwiseOperator(const CuMat &mat)
    : Mat(mat) { }

CuMat CuRowwiseOperator::Sum() const
{
    return Sum(CuIdentity());
}

CuMat CuColwiseOperator::Sum() const
{
    return Sum(CuIdentity());
}

