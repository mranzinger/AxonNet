/*
 * test_helper.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#include "test_helper.h"

#include <iostream>

#include <gtest/gtest.h>

const Real DEFAULT_PRECISION = 0.001;

using namespace std;

namespace {

template<typename MatTypeA, typename MatTypeB>
void AssertEquivalence(const MatTypeA &a, const MatTypeB &b, Real precision)
{
	ASSERT_EQ(a.rows(), b.rows());
	ASSERT_EQ(a.cols(), b.cols());

	for (int row = 0; row < a.rows(); ++row)
	{
		for (int col = 0; col < a.cols(); ++col)
		{
			const Real aVal = a(row, col);
			const Real bVal = b(row, col);

			if (abs(aVal - bVal) > precision)
			{
				cout << "[" << row << ", " << col << "] "
				     << aVal << " != " << bVal << endl;
				ASSERT_TRUE(false);
			}
		}
	}
}

}

void AssertVectorEquivalence(const Vector& a, const Vector& b, Real precision)
{
	AssertEquivalence(a, b, precision);
}

void AssertMatrixEquivalence(const CMatrix& a, const CMatrix& b,
		Real precision)
{
    AssertEquivalence(a, b, precision);
}

void AssertMatrixEquivalence(const RMatrix& a, const RMatrix& b,
		Real precision)
{
	AssertEquivalence(a, b, precision);
}
