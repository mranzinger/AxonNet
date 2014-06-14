/*
 * test_helper.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#include "test_helper.h"

#include <iostream>

#include <gtest/gtest.h>

const Real DEFAULT_PRECISION = 0.0001;

using namespace std;

namespace {

template<typename MatTypeA, typename MatTypeB>
void AssertEquivalence(const MatTypeA &a, const MatTypeB &b, Real precision)
{
    bool eq = a.isApprox(b, precision);

    if (!eq)
    {
        if (a.size() < 100 && b.size() < 100)
        {
            cout << "Matrix A:" << endl << a << endl << endl
                 << "Matrix B:" << endl << b << endl << endl;
        }
        // Obviously, this is false... do it for the debug statement
        ASSERT_TRUE(a.isApprox(b, precision));
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
