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

TEST(CuMatTest, Copy)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat cMat(handle, 3, 3);

    CMatrix mat(3, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;

    cMat.CopyToDevice(mat);

    CMatrix matB(3, 3);
    cMat.CopyToHost(matB);

    AssertMatrixEquivalence(mat, matB);
}

TEST(CuMatTest, SetConstant)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat a(handle, 3, 3);

    CMatrix hostMat(3, 3);
    hostMat << 5, 33, 1,
               2, 3,  7,
               11, 12, 99;

    a.CopyToDevice(hostMat);

    a.SetConstant(42);

    a.CopyToHost(hostMat);

    CMatrix correctMat(3, 3);
    correctMat << 42, 42, 42,
                  42, 42, 42,
                  42, 42, 42;

    AssertMatrixEquivalence(hostMat, correctMat);
}

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

	CMatrix aCopy(3, 3), bCopy(3, 3);
	a.CopyToHost(aCopy);
	b.CopyToHost(bCopy);

	AssertMatrixEquivalence(cMatA, aCopy);
	AssertMatrixEquivalence(cMatB, bCopy);

	AssertMatrixEquivalence(cMatC, comp);
}


