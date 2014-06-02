/*
 * File description: cumat_add.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <assert.h>

using namespace std;

CuMat operator+(const CuMat &a, const CuMat &b)
{
    CuMat ret;
    a.BinaryExpr<false>(b, ret, CuPlus());
    return ret;
}
CuMat operator-(const CuMat &a, const CuMat &b)
{
    CuMat ret;
    a.BinaryExpr<false>(b, ret, CuMinus());
    return ret;
}


CuMat &operator+=(CuMat &a, const CuMat &b)
{
    a.BinaryExpr(b, CuPlus());
    return a;
}
CuMat &operator-=(CuMat &a, const CuMat &b)
{
    a.BinaryExpr(b, CuMinus());
    return a;
}

void CuMat::AddScaled(Real scaleThis, const CuMat& b, Real scaleB)
{
    AddScaled(scaleThis, b, scaleB, *this);
}

void CuMat::AddScaled(Real scaleThis, const CuMat& b, Real scaleB,
        CuMat& dest) const
{
    AssertSameDims(b);

    BinaryExpr<false>(b, dest, CuAddScaledBinary(scaleThis, scaleB));
}

void AddScaled(const CuMat &a, Real scaleA, const CuMat &b, Real scaleB, CuMat &dest)
{
    a.AddScaled(scaleA, b, scaleB, dest);
}
