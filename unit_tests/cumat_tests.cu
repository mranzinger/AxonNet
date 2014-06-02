/*
 * cumat_tests.cu
 *
 *  Created on: May 30, 2014
 *      Author: mike
 */

#include <gtest/gtest.h>

#include "cumat.cuh"

#include "inc/cublas_ut_helper.cuh"
#include "test_helper.h"

TEST(CuMatTest, Add)
{
	cublasHandle_t handle = UTGetCublasHandle();

	CuMat a(handle, 3, 3),
		  b(handle, 3, 3),
		  c;

	CMatrix cMatA(3, 3), cMatB(3, 3);
	cMatA << 1, 2, 3,
			 4, 5, 6,
			 7, 8, 9;
	cMatB << 9, 8, 7,
			 6, 5, 4,
			 3, 2, 1;

	CMatrix cMatC(3, 3);
	cMatC << 10, 10, 10,
			 10, 10, 10,
			 10, 10, 10;

	a.CopyToDevice(cMatA);
	b.CopyToDevice(cMatB);

	c = a + b;

	CMatrix comp(3, 3);
	c.CopyToHost(comp);

	AssertMatrixEquivalence(cMatC, comp);
}


