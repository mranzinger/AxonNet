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

TEST(CuMatTest, Copy2)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dMat(handle, 2, 4);

    CMatrix hMat(2, 4);
    hMat << 1, 1, 1, 1,
            2, 2, 2, 2;

    dMat.CopyToDevice(hMat);

    CMatrix hCorrect;
    dMat.CopyToHost(hCorrect);

    AssertMatrixEquivalence(hMat, hCorrect);
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

TEST(CuMatTest, SetConstant2)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dMat(handle, 40, 20);
    dMat = -1;

    CMatrix hMat;
    dMat.CopyToHost(hMat);

    CMatrix hCorrect(40, 20);
    hCorrect.setConstant(-1);

    AssertMatrixEquivalence(hMat, hCorrect);
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

TEST(CuMatTest, Add2)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 10, 5),
          dB(handle, 10, 5);

    dA.SetConstant(1);
    dB.SetConstant(-1);

    dA += dB;

    CMatrix hMat;
    dA.CopyToHost(hMat);

    CMatrix hCorrect = CMatrix::Zero(10, 5);

    AssertMatrixEquivalence(hMat, hCorrect);
}

TEST(CuMatTest, Mul)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 3, 3),
          dB(handle, 3, 3),
          dC;

    CMatrix hID = CMatrix::Identity(3, 3);
    dA.CopyToDevice(hID);

    CMatrix hOther(3, 3);
    hOther << 1, 2, 3,
              4, 5, 6,
              7, 8, 9;

    dB.CopyToDevice(hOther);

    dC = dA * dB;

    CMatrix hComp;
    dC.CopyToHost(hComp);

    AssertMatrixEquivalence(hComp, hOther);
}

TEST(CuMatTest, Mul2)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 1000, 2000),
          dB(handle, 2000, 128);

    CMatrix hA = CMatrix::Random(1000, 2000),
            hB = CMatrix::Random(2000, 128);

    dA.CopyToDevice(hA);
    dB.CopyToDevice(hB);

    CMatrix hC = hA * hB;

    CuMat dC = dA * dB;

    CMatrix hComp;
    dC.CopyToHost(hComp);

    AssertMatrixEquivalence(hComp, hC);
}

TEST(CuMatTest, MulEigen)
{
    CMatrix hA = CMatrix::Random(1000, 2000),
            hB = CMatrix::Random(2000, 128);

    CMatrix hC = hA * hB;
}

TEST(CuMatTest, MulCuda)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 1000, 2000),
          dB(handle, 2000, 128);

    CMatrix hA = CMatrix::Random(1000, 2000),
            hB = CMatrix::Random(2000, 128);

    dA.CopyToDevice(hA);
    dB.CopyToDevice(hB);

    CuMat dC = dA * dB;
}

TEST(CuMatTest, MulEigenHuge)
{
    CMatrix hA = CMatrix::Constant(10000, 20000, 2),
            hB = CMatrix::Constant(20000, 128, 4);

    CMatrix hC = hA * hB;
}

TEST(CuMatTest, MulCudaHuge)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 10000, 20000),
          dB(handle, 20000, 128);

    dA.SetConstant(2);
    dB.SetConstant(4);

    CuMat dC = dA * dB;
}

TEST(CuMatTest, AddScaled)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 2, 3),
          dB(handle, 2, 3),
          dC;

    dA.SetConstant(-2);
    dB.SetConstant(1);

    AddScaled(dA, 1, dB, 2, dC);

    CMatrix hMat;
    dC.CopyToHost(hMat);

    CMatrix hCorrect = CMatrix::Zero(2, 3);

    AssertMatrixEquivalence(hMat, hCorrect);
}


