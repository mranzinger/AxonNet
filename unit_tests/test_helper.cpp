/*
 * test_helper.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#include "test_helper.h"

#include <gtest/gtest.h>

const Real DEFAULT_PRECISION = 0.0001;

void AssertVectorEquivalence(const Vector& a, const Vector& b, Real precision)
{
	ASSERT_TRUE(a.isApprox(b, precision));
}

void AssertMatrixEquivalence(const CMatrix& a, const CMatrix& b,
		Real precision)
{
	ASSERT_TRUE(a.isApprox(b, precision));
}

void AssertMatrixEquivalence(const RMatrix& a, const RMatrix& b,
		Real precision)
{
	ASSERT_TRUE(a.isApprox(b, precision));
}
