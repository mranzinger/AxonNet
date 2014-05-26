#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>

using namespace std;

CuMat::CuMat()
	: _handle(0), _refCt(NULL), _dMat(NULL), _rows(0), _cols(0), _storageOrder(CuColMajor)
{
}

CuMat::CuMat(cublasHandle_t handle,
		     unsigned long rows, unsigned long cols, 
		     CuStorageOrder order)
	: _handle(handle), _dMat(NULL), _rows(rows), _cols(cols), _storageOrder(order)
{
	_refCt = new int(1);
	
	cudaError_t cudaStat = cudaMalloc(&_dMat, rows * cols * sizeof(float));
	if (cudaStat != cudaSuccess)
		throw runtime_error("Unable to allocate the specified matrix");
}

CuMat::CuMat(const CuMat &other)
	: _handle(other._handle), _dMat(other._dMat), _rows(other._rows), _cols(other._cols),
	  _refCt(other._refCt), _storageOrder(other._storageOrder)
{
	// Increment the ref count
	++(*_refCt);
}

CuMat::~CuMat()
{
	// Decrement the ref count
	--(*_refCt);
	if (*_refCt == 0)
	{
		// Free the device memory
		cudaFree(_dMat);
	}
}

void swap(CuMat &a, CuMat &b)
{
	swap(a._handle, b._handle);
	swap(a._dMat, b._dMat);
	swap(a._rows, b._rows);
	swap(a._cols, b._cols);
	swap(a._refCt, b._refCt);
	swap(a._storageOrder, b._storageOrder);
}