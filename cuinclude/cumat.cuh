/*
 * cumat.h
 *
 *  Created on: May 25, 2014
 *      Author: mike
 */


#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>

#include <Eigen/Dense>

#include "math_defines.h"

enum CuStorageOrder
{
	CuRowMajor,
	CuColMajor
};

class CuMat
{
public:
	CuMat();
	CuMat(cublasHandle_t handle, unsigned long rows, unsigned long cols, 
		  CuStorageOrder order = CuColMajor);
	CuMat(const CuMat &other);
	~CuMat();
	CuMat &operator=(CuMat other);
	CuMat Copy() const;
	
	void CopyToDevice(const Real *hMatrix);
	void CopyToDevice(const CMatrix &hMatrix);
	void CopyToDevice(const RMatrix &hMatrix);
	void CopyToDeviceAsync(const Real *hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const CMatrix &hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const RMatrix &hMatrix, cudaStream_t stream);
	
	void CopyToHost(Real *hMatrix) const;
	void CopyToHost(CMatrix &hMatrix) const;
	void CopyToHost(RMatrix &hMatrix) const;
	void CopyToHostAsync(Real *hMatrix, cudaStream_t stream);
	void CopyToHostAsync(CMatrix &hMatrix, cudaStream_t stream);

	friend void swap(CuMat &a, CuMat &b);
	
private:
	Real *_dMat;
	int *_refCt;
	unsigned long _rows, _cols;
	CuStorageOrder _storageOrder;
	cublasHandle_t _handle;
};
