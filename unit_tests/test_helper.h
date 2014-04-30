/*
 * test_helper.h
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#pragma once

#include "math_util.h"

extern const Real DEFAULT_PRECISION;

void AssertVectorEquivalence(const Vector &a, const Vector &b, Real precision = DEFAULT_PRECISION);
void AssertMatrixEquivalence(const CMatrix &a, const CMatrix &b, Real precision = DEFAULT_PRECISION);
void AssertMatrixEquivalence(const RMatrix &a, const RMatrix &b, Real precision = DEFAULT_PRECISION);
