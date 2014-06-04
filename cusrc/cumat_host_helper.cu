/*
 * File description: cumat_host_helper.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat_host_helper.h"
#include "cumat.cuh"

uint32_t CuMat_Rows(CuMat *m)
{
    return m->Rows();
}

uint32_t CuMat_Cols(CuMat *m)
{
    return m->Cols();
}

void CuMat_Delete(CuMat *m)
{
    delete m;
}

void CuMat_SafeDelete(CuMat *&m)
{
    delete m;
    m = NULL;
}

CuMat* CuMat_CopyToDevice(const CMatrix& hMat, cublasHandle_t handle)
{
    CuMat *m = new CuMat(handle, hMat.rows(), hMat.cols());
    m->CopyToDevice(hMat);
    return m;
}

void CuMat_CopyToDevice(const CMatrix& hMat, CuMat& dMat)
{
    dMat.CopyToDevice(hMat);
}

CMatrix* CuMat_CopyToHost(const CuMat &dMat)
{
    CMatrix *ret = new CMatrix(dMat.Rows(), dMat.Cols());
    dMat.CopyToHost(*ret);
    return ret;
}

void CuMat_CopyToHost(const CuMat &dMat, CMatrix &hMat)
{
    dMat.CopyToHost(hMat);
}
