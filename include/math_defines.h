/*
 * math_defines.h
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#include <Eigen/Dense>
#include <vector>
#include <stdint.h>

typedef float Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Real, 1, Eigen::Dynamic> RowVector;
typedef std::vector<Vector> MultiVector;
typedef Eigen::Map<Vector, Eigen::Aligned> MapVector;
typedef Eigen::Map<RowVector, Eigen::Aligned> MapRowVector;
typedef Eigen::Map<Vector, Eigen::Unaligned> UMapVector;
typedef Eigen::Map<RowVector, Eigen::Unaligned> UMapRowVector;
typedef Eigen::Map<Vector, 0, Eigen::OuterStride<> > StrideVec;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> CMatrix;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;

typedef Eigen::Map<CMatrix> CMap;
typedef Eigen::Map<RMatrix> RMap;

typedef Eigen::Map<CMatrix, Eigen::Unaligned> CUMap;
typedef Eigen::Map<RMatrix, Eigen::Unaligned> RUMap;

typedef Eigen::Map<CMatrix, 0, Eigen::OuterStride<> > CStrideMap;
typedef Eigen::Map<RMatrix, 0, Eigen::OuterStride<> > RStrideMap;

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

struct CuContext
{
	int Device;
	cublasHandle_t CublasHandle;

	CuContext() : Device(0), CublasHandle(0) { }
	explicit CuContext(cublasHandle_t handle) : Device(0), CublasHandle(handle) { }
	CuContext(int device, cublasHandle_t handle) : Device(device), CublasHandle(handle) { }

	bool operator==(const CuContext &b) const
	{
		return Device == b.Device &&
			   CublasHandle == b.CublasHandle;
	}
	bool operator!=(const CuContext &b) const
	{
		return !(*this == b);
	}

	cudaStream_t GetStream() const;
	void SetStream(cudaStream_t stream);
};
