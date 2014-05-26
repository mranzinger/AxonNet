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
	
	friend void swap(CuMat &a, CuMat &b);
	
private:
	float *_dMat;
	int *_refCt;
	unsigned long _rows, _cols;
	CuStorageOrder _storageOrder;
	cublasHandle_t _handle;
};
