/*
 * File description: cumat_agg.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cumat.cuh"

#include <stdexcept>

using namespace std;

CuRowwiseOperator::CuRowwiseOperator(const CuMat &mat)
    : Mat(mat) { }

CuColwiseOperator::CuColwiseOperator(const CuMat &mat)
    : Mat(mat) { }

CuMat CuRowwiseOperator::Sum() const
{
    return Sum(CuIdentity());
}

void CuRowwiseOperator::Sum(CuMat &dest, cublasHandle_t cublasHandle) const
{
	Sum(dest, CuIdentity(), cublasHandle);
}

CuMat CuColwiseOperator::Sum() const
{
    return Sum(CuIdentity());
}

void CuColwiseOperator::Sum(CuMat &dest, cublasHandle_t cublasHandle) const
{
	Sum(dest, CuIdentity(), cublasHandle);
}

CuMat CuRowwiseOperator::Max() const
{
	return Max(CuIdentity());
}

CuMat CuColwiseOperator::Max() const
{
	return Max(CuIdentity());
}

CuMat CuRowwiseOperator::Min() const
{
	return Min(CuIdentity());
}

CuMat CuColwiseOperator::Min() const
{
	return Min(CuIdentity());
}

CuMat CuRowwiseOperator::MaxIdx() const
{
	CuMat ret(Mat.Handle(), Mat.Rows(), 1, Mat.Order());

	CMatrix hMat(Mat.Rows(), 1);

	for (uint32_t row = 0; row < Mat.Rows(); ++row)
	{
		int idx;
		cublasStatus_t status;

		if (Mat.Order() == CuColMajor)
		{
			status = cublasIsamax_v2(Mat.Handle().CublasHandle,
									 Mat.Cols(),
									 Mat._dMat + row,
									 Mat.Rows(), &idx);
		}
		else
		{
			status = cublasIsamax_v2(Mat.Handle().CublasHandle,
									 Mat.Cols(),
									 Mat._dMat + row * Mat.Cols(),
									 1,
									 &idx);
		}

		if (status != CUBLAS_STATUS_SUCCESS)
			throw runtime_error("Unable to compute the max element for this row.");

		hMat(row, 0) = idx - 1;
	}

	ret.CopyToDevice(hMat);

	return ret;
}

CuMat CuColwiseOperator::MaxIdx() const
{
	CuMat ret(Mat.Handle(), 1, Mat.Cols(), Mat.Order());

	CMatrix hMat(1, Mat.Cols());

	for (uint32_t col = 0; col < Mat.Cols(); ++col)
	{
		int idx;
		cublasStatus_t status;

		if (Mat.Order() == CuColMajor)
		{
			status = cublasIsamax_v2(Mat.Handle().CublasHandle,
								     Mat.Rows(),
								     Mat._dMat + col * Mat.Rows(),
								     1,
								     &idx);
		}
		else
		{
			status = cublasIsamax_v2(Mat.Handle().CublasHandle,
									 Mat.Rows(),
									 Mat._dMat + col,
									 Mat.Cols(),
									 &idx);
		}

		if (status != CUBLAS_STATUS_SUCCESS)
			throw runtime_error("Unable to compute the min element for this row.");

		hMat(0, col) = idx - 1;
	}

	ret.CopyToDevice(hMat);

	return ret;
}

CuMat CuRowwiseOperator::MinIdx() const
{
	CuMat ret(Mat.Handle(), Mat.Rows(), 1, Mat.Order());

	Vector hMat(Mat.Rows());

	for (uint32_t row = 0; row < Mat.Rows(); ++row)
	{
		int idx;
		cublasStatus_t status;

		if (Mat.Order() == CuColMajor)
		{
			status = cublasIsamin_v2(Mat.Handle().CublasHandle,
									 Mat.Cols(),
									 Mat._dMat + row,
									 Mat.Rows(), &idx);
		}
		else
		{
			status = cublasIsamin_v2(Mat.Handle().CublasHandle,
									 Mat.Cols(),
									 Mat._dMat + row * Mat.Cols(),
									 1,
									 &idx);
		}

		if (status != CUBLAS_STATUS_SUCCESS)
			throw runtime_error("Unable to compute the max element for this row.");

		hMat(row) = idx - 1;
	}

	ret.CopyToDevice(hMat);

	return ret;
}

CuMat CuColwiseOperator::MinIdx() const
{
	CuMat ret(Mat.Handle(), 1, Mat.Cols(), Mat.Order());

	CMatrix hMat(1, Mat.Cols());

	for (uint32_t col = 0; col < Mat.Cols(); ++col)
	{
		int idx;
		cublasStatus_t status;

		if (Mat.Order() == CuColMajor)
		{
			status = cublasIsamin_v2(Mat.Handle().CublasHandle,
								     Mat.Rows(),
								     Mat._dMat + col * Mat.Rows(),
								     1,
								     &idx);
		}
		else
		{
			status = cublasIsamin_v2(Mat.Handle().CublasHandle,
									 Mat.Rows(),
									 Mat._dMat + col,
									 Mat.Cols(),
									 &idx);
		}

		if (status != CUBLAS_STATUS_SUCCESS)
			throw runtime_error("Unable to compute the min element for this row.");

		hMat(0, col) = idx - 1;
	}

	ret.CopyToDevice(hMat);

	return ret;
}
