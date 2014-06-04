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
