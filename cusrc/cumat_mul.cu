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
    return ScaledMultiply(1.0f, a, b, 0);
}

CuMat operator*(const CuScopedWeakTranspose &tA, const CuMat &b)
{
    return ScaledMultiply(1.0f, tA, b, 0);
}

CuMat operator*(const CuMat &a, const CuScopedWeakTranspose &tB)
{
    return ScaledMultiply(1.0f, a, tB, 0);
}

CuMat operator*(const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB)
{
    return ScaledMultiply(1.0f, tA, tB, 0);
}


CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b, cublasHandle_t cublasHandle)
{
    CuMat dest(a._handle);

    ScaledMultiply(scale, a, b, 0.0f, dest, cublasHandle);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b, cublasHandle_t cublasHandle)
{
    CuMat dest(b._handle);

    ScaledMultiply(scale, tA, b, 0.0f, dest, cublasHandle);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle)
{
    CuMat dest(a._handle);

    ScaledMultiply(scale, a, tB, 0.0f, dest, cublasHandle);

    return dest;
}
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle)
{
    CuMat dest(tA.Mat._handle);

    ScaledMultiply(scale, tA, tB, 0.0f, dest, cublasHandle);

    return dest;
}

CuMat MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols, const CuMat &b)
{
	CuMat dest(a._handle);

	MultiplyTrans3D(a, rows, cols, b, dest);

	return dest;
}

void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle)
{
    assert(!a.Empty() && !b.Empty());
    // Make sure the matrices are valid
    assert(a._cols == b._rows);

    // TODO: Support these other cases
    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (!cublasHandle)
    	cublasHandle = a._handle.CublasHandle;

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._rows, b._cols);
    }
    else
    {
        assert(dest._rows == a._rows && dest._cols == b._cols);
    }

    cublasStatus_t status =
            cublasSgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            a._rows, b._cols, a._cols,
                            &mulScale, a._dMat, a._rows,
                            b._dMat, b._rows,
                            &scaleDest,
                            dest._dMat,
                            dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}

void MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols, const CuMat &b, CuMat &dest)
{
	assert(!a.Empty() && !b.Empty());

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    assert(cols == b._rows);
    assert(a._rows == (rows * cols));

    dest.Resize(rows, b._cols);

    const float alpha = 1.0f, beta = 0.0f;

    for (uint32_t i = 0; i < b._cols; ++i)
    {
        cublasStatus_t status =
                cublasSgemm_v2(a._handle.CublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                               rows, 1, cols,
                               &alpha,
                               a._dMat + i * a._rows,
                               rows,
                               b._dMat + i * b._rows,
                               b._rows,
                               &beta,
                               dest._dMat + i * rows,
                               rows);

        if (status != CUBLAS_STATUS_SUCCESS)
            throw runtime_error("The matrix multiplication failed.");
    }
}

void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA,
                    const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle)
{
    const CuMat &a = tA.Mat;

    assert(a._rows == b._rows);
    assert(!a.Empty() && !b.Empty());

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (!cublasHandle)
    	cublasHandle = a._handle.CublasHandle;

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._cols, b._cols);
    }
    else
    {
        assert(dest._rows == a._cols && dest._cols == b._cols);
    }

    cublasStatus_t status =
            cublasSgemm_v2(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
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
                    const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle)
{
    const CuMat &b = tB.Mat;

    assert(a._cols == b._cols);
    assert(!a.Empty() && !b.Empty());

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (!cublasHandle)
    	cublasHandle = a._handle.CublasHandle;

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._rows, b._rows);
    }
    else
    {
        assert(dest._rows == a._rows && dest._cols == b._rows);
    }

    cublasStatus_t status =
            cublasSgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
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
                    const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle)
{
    const CuMat &a = tA.Mat;
    const CuMat &b = tB.Mat;

    assert(a._rows == b._cols);
    assert(!a.Empty() && !b.Empty());

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    if (!cublasHandle)
    	cublasHandle = a._handle.CublasHandle;

    if (scaleDest == 0.0f)
    {
        dest.Resize(a._cols, b._rows);
    }
    else
    {
        assert(dest._rows == a._cols && dest._cols == b._rows);
    }

    cublasStatus_t status =
            cublasSgemm_v2(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                    a._cols, b._rows, a._rows,
                    &mulScale, a._dMat, a._rows,
                    b._dMat, b._rows,
                    &scaleDest,
                    dest._dMat,
                    dest._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");
}




