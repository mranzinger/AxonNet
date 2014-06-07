/*
 * File description: cumat_host_helper.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "math_util.h"

class CuMat;

uint32_t CuMat_Rows(CuMat *m);
uint32_t CuMat_Cols(CuMat *m);

void CuMat_Delete(CuMat *m);
void CuMat_SafeDelete(CuMat *&m);

CuMat *CuMat_CopyToDevice(const CMatrix &hMat, CuContext handle);
void CuMat_CopyToDevice(const CMatrix &hMat, CuMat &dMat);

CMatrix *CuMat_CopyToHost(const CuMat &dMat);
void CuMat_CopyToHost(const CuMat &dMat, CMatrix &hMat);

CuMat *CuMat_Make(const CuContext &handle, uint32_t rows, uint32_t cols);

CuMat *CuMat_MakeSimilar(const CuMat &m);
CuMat *CuMat_MakeSimilar(const CuContext &handle, const CuMat &m);
