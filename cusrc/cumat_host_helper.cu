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

