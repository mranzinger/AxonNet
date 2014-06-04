/*
 * File description: cumat_scalar.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat.cuh"

CuMat &CuMat::operator *=(Real val)
{
    CoeffMultiply(val);
    return *this;
}
CuMat &CuMat::operator /=(Real val)
{
    return *this *= (1.0f / val);
}

CuMat operator*(const CuMat &m, Real scale)
{
    CuMat ret;
    m.CoeffMultiply(scale, ret);
    return ret;
}
CuMat operator*(Real scale, const CuMat &m)
{
    CuMat ret;
    m.CoeffMultiply(scale, ret);
    return ret;
}
CuMat operator/(const CuMat &m, Real scale)
{
    CuMat ret;
    m.CoeffMultiply(1.0f / scale, ret);
    return ret;
}



CuMat operator+(const CuMat &m, Real val)
{
    CuMat ret;
    m.UnaryExpr<false>(ret, CuScalarAdd(val));
    return ret;
}
CuMat operator+(Real val, const CuMat &m)
{
    CuMat ret;
    m.UnaryExpr<false>(ret, CuScalarAdd(val));
    return ret;
}
CuMat operator-(const CuMat &m, Real val)
{
    CuMat ret;
    m.UnaryExpr<false>(ret, CuScalarAdd(-val));
    return ret;
}

struct CuSubRight
{
    Real _val;

    CuSubRight(Real val) : _val(val) { }

    __device__ Real operator()(Real mVal) const
    {
        return _val - mVal;
    }
};

CuMat operator-(Real val, const CuMat &m)
{
    CuMat ret;
    m.UnaryExpr<false>(ret, CuSubRight(val));
    return ret;
}
