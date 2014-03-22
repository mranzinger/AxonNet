#pragma once

#include "CppUnitTest.h"
#include "math_util.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetTest
{
	const Real DEFAULT_PRECISION = 0.0001;

	void AssertVectorEquivalence(const Vector &a, const Vector &b, Real precision = DEFAULT_PRECISION)
	{
		Assert::IsTrue(a.isApprox(b, precision));
	}
	void AssertMatrixEquivalence(const Matrix &a, const Matrix &b, Real precision = DEFAULT_PRECISION)
	{
		Assert::IsTrue(a.isApprox(b, precision));
	}
}