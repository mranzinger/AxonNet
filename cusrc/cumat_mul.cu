/*
 * File description: cumat_mul.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <assert.h>

using namespace std;

CuMat operator*(const CuMat &a, const CuMat &b)
{
    return ScaledMultiply(1.0f, a, b);
}

CuMat operator*(const CuScopedWeakTranspose &tA, const CuMat &b)
{
    return ScaledMultiply(1.0f, tA, b);
}

CuMat operator*(const CuMat &a, const CuScopedWeakTranspose &tB)
{
    return ScaledMultiply(1.0f, a, tB);
}

CuMat operator*(const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB)
{
    return ScaledMultiply(1.0f, tA, tB);
}


CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b)
{
    CuMat dest(a._handle);

    ScaledMultiply(scale, a, b, 0.0f, dest);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b)
{
    CuMat dest(b._handle);

    ScaledMultiply(scale, tA, b, 0.0f, dest);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB)
{
    CuMat dest(a._handle);

    ScaledMultiply(scale, a, tB, 0.0f, dest);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB)
{
    CuMat dest(tA.Mat._handle);

    ScaledMultiply(scale, tA, tB, 0.0f, dest);

    return dest;
}

void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest)
{
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);
    // Make sure the matrices are valid
    assert(a._cols == b._rows);

    // TODO: Support these other cases
    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._rows, b._cols);
    }
    else
    {
        assert(dest._rows == a._rows && dest._cols == b._cols);
    }

    cublasStatus_t status =
            cublasSgemm_v2(a._handle.CublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            a._rows, b._cols, a._cols,
                            &mulScale, a._dMat, a._rows,
                            b._dMat, b._rows,
                            &scaleDest,
                            dest._dMat,
                            dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}
void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA,
                    const CuMat &b, Real scaleDest, CuMat &dest)
{
    const CuMat &a = tA.Mat;

    assert(a._rows == b._rows);
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._cols, b._cols);
    }
    else
    {
        assert(dest._rows == a._cols && dest._cols == b._cols);
    }

    cublasStatus_t status =
            cublasSgemm_v2(a._handle.CublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                           a._cols, b._cols, a._rows,
                           &mulScale, a._dMat, a._rows,
                           b._dMat, b._rows,
                           &scaleDest,
                           dest._dMat,
                           dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}
void ScaledMultiply(Real mulScale, const CuMat &a,
                    const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest)
{
    const CuMat &b = tB.Mat;

    assert(a._cols == b._cols);
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._rows, b._rows);
    }
    else
    {
        assert(dest._rows == a._rows && dest._cols == b._rows);
    }

    cublasStatus_t status =
            cublasSgemm_v2(a._handle.CublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                           a._rows, b._rows, a._cols,
                           &mulScale, a._dMat, a._rows,
                           b._dMat, b._rows,
                           &scaleDest,
                           dest._dMat,
                           dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}



void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA,
                    const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest)
{
    const CuMat &a = tA.Mat;
    const CuMat &b = tB.Mat;

    assert(a._rows == b._cols);
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._cols, b._rows);
    }
    else
    {
        assert(dest._rows == a._cols && dest._cols == b._rows);
    }

    cublasStatus_t status =
            cublasSgemm_v2(a._handle.CublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                    a._cols, b._rows, a._rows,
                    &mulScale, a._dMat, a._rows,
                    b._dMat, b._rows,
                    &scaleDest,
                    dest._dMat,
                    dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}


