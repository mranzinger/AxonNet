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

TEST(CuMatTest, Add3)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 10, 5),
          dB(handle,  5, 10);

    dA.SetConstant(1);
    dB.SetConstant(-1);

    dA += dB.Transpose();

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

TEST(CuMatTest, Mul3)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 2000, 1000),
          dB(handle, 2000, 128);

    CMatrix hA = CMatrix::Random(2000, 1000),
            hB = CMatrix::Random(2000, 128);

    dA.CopyToDevice(hA);
    dB.CopyToDevice(hB);

    CMatrix hC = hA.transpose() * hB;

    CuMat dC = dA.WeakTranspose() * dB;

    CMatrix hComp;
    dC.CopyToHost(hComp);

    AssertMatrixEquivalence(hComp, hC);
}

TEST(CuMatTest, Mul4)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 1000, 2000),
          dB(handle, 128, 2000);

    CMatrix hA = CMatrix::Random(1000, 2000),
            hB = CMatrix::Random(128, 2000);

    dA.CopyToDevice(hA);
    dB.CopyToDevice(hB);

    CMatrix hC = hA * hB.transpose();

    CuMat dC = dA * dB.WeakTranspose();

    CMatrix hComp;
    dC.CopyToHost(hComp);

    AssertMatrixEquivalence(hComp, hC);
}

TEST(CuMatTest, Mul5)
{
    cublasHandle_t handle = UTGetCublasHandle();

    CuMat dA(handle, 2000, 1000),
          dB(handle, 128, 2000);

    CMatrix hA = CMatrix::Random(2000, 1000),
            hB = CMatrix::Random(128, 2000);

    dA.CopyToDevice(hA);
    dB.CopyToDevice(hB);

    CMatrix hC = hA.transpose() * hB.transpose();

    CuMat dC = dA.WeakTranspose() * dB.WeakTranspose();

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

/*TEST(CuMatTest, MulEigenHuge)
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
}*/

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

TEST(CuMatTest, SumColumns)
{
	cublasHandle_t handle = UTGetCublasHandle();

	CuMat dA(handle, 1000, 128);
	dA.SetConstant(1);

	CuMat dComp = dA.Rowwise().Sum();

	ASSERT_EQ(dComp.Rows(), dA.Rows());
	ASSERT_EQ(dComp.Cols(), 1);

	CMatrix hComp;
	dComp.CopyToHost(hComp);

	CMatrix hCorrect = CMatrix::Constant(1000, 128, 1).rowwise().sum();

	AssertMatrixEquivalence(hComp, hCorrect);
}

struct HExp
{
	Real operator()(Real val) const { return exp(val); }
};

TEST(CuMatTest, SumRows)
{
	cublasHandle_t handle = UTGetCublasHandle();

	CuMat dA(handle, 1000, 128);
	dA.SetConstant(0.95f);

	// Sum the exponentials
	CuMat dComp = dA.Colwise().Sum(CuExp());

	ASSERT_EQ(dComp.Rows(), 1);
	ASSERT_EQ(dComp.Cols(), dA.Cols());

	CMatrix hComp;
	dComp.CopyToHost(hComp);

	CMatrix hCorrect = CMatrix::Constant(1000, 128, 0.95f).unaryExpr(HExp()).colwise().sum();

	AssertMatrixEquivalence(hComp, hCorrect);
}

struct EigSoftmaxExpr
{
	Real operator()(Real value, Real colMax) const
	{
		return exp(value - colMax);
	}
};

struct EigDiv
{
	Real operator()(Real numerator, Real denomonator) const
	{
		return numerator / denomonator;
	}
};

struct CuSoftmaxExpr
{
	const Real *_maxBuff;

	CuSoftmaxExpr(const CuMat &mat)
		: _maxBuff(mat.Buff()) { }

	Real operator()(Real value, uint32_t row, uint32_t col) const
	{
		return exp(value - _maxBuff[col]);
	}
};

TEST(CuMatTest, Softmax)
{
	cublasHandle_t handle = UTGetCublasHandle();

	CuMat dInput(handle, 1000, 128);

	CMatrix hInput = CMatrix::Random(1000, 128);

	dInput.CopyToDevice(hInput);

	// Host Computation
	// Get the maximum value of each column
	CMatrix hIpMax = hInput.colwise().maxCoeff().replicate(1000, 1);

	ASSERT_EQ(hIpMax.rows(), 1000);
	ASSERT_EQ(hIpMax.cols(), 128);

	CMatrix hExpMat = hInput.binaryExpr(hIpMax, EigSoftmaxExpr());

	CMatrix hExpMatSum = hExpMat.colwise().sum().replicate(1000, 1);

	CMatrix hSoftmax = hExpMat.binaryExpr(hExpMatSum, EigDiv());

	// Device Computation
	CuMat dIpMax = dInput.Colwise().Max();


}




























