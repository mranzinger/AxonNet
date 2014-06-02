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

namespace {
    const float s_one = 1.0f;
    const float s_zero = 0.0f;
}

CuMat operator*(const CuMat &a, const CuMat &b)
{
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);
    // Make sure the matrices are valid
    assert(a._cols == b._rows);

    // TODO: Support these other cases
    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    CuMat ret(a._handle, a._rows, b._cols);

    cublasStatus_t status =
            cublasSgemm_v2(a._handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            a._rows, b._cols, a._cols,
                            &s_one, a._dMat, a._rows,
                            b._dMat, b._rows,
                            &s_zero,
                            ret._dMat,
                            ret._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");

    return ret;
}

CuMat operator*(const CuScopedWeakTranspose &tA, const CuMat &b)
{
    const CuMat &a = tA.Mat;

    assert(a._rows == b._rows);
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    CuMat ret(a._handle, a._cols, b._cols);

    cublasStatus_t status =
            cublasSgemm_v2(a._handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           a._cols, b._cols, a._rows,
                           &s_one, a._dMat, a._rows,
                           b._dMat, b._rows,
                           &s_zero,
                           ret._dMat,
                           ret._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");

    return ret;
}

CuMat operator*(const CuMat &a, const CuScopedWeakTranspose &tB)
{
    const CuMat &b = tB.Mat;

    assert(a._cols == b._cols);
    assert(!a.Empty() && !b.Empty());
    assert(a._handle == b._handle);

    assert(a._storageOrder == CuColMajor && b._storageOrder == CuColMajor);

    CuMat ret(a._handle, a._rows, b._rows);

    cublasStatus_t status =
            cublasSgemm_v2(a._handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           a._rows, b._rows, a._cols,
                           &s_one, a._dMat, a._rows,
                           b._dMat, b._rows,
                           &s_zero,
                           ret._dMat,
                           ret._rows);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw runtime_error("The matrix multiplication failed.");

    return ret;
}

CuMat operator*(const CuScopedWeakTranspose &a, const CuScopedWeakTranspose &b)
{
    throw runtime_error("Not implemented");
}

