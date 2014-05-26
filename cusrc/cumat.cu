#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <assert.h>

using namespace std;

CuMat::CuMat()
	: _handle(0), _refCt(NULL), _dMat(NULL), _rows(0), _cols(0), _storageOrder(CuColMajor)
{
	_refCt = new int(1);
}

CuMat::CuMat(cublasHandle_t handle,
		     unsigned long rows, unsigned long cols, 
		     CuStorageOrder order)
	: _handle(handle), _dMat(NULL), _rows(rows), _cols(cols), _storageOrder(order)
{
	_refCt = new int(1);
	
	cudaError_t cudaStat = cudaMalloc(&_dMat, rows * cols * sizeof(Real));
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
		delete _refCt;
		// Free the device memory
		cudaFree(_dMat);
	}
}

CuMat &CuMat::operator=(CuMat other)
{
	swap(*this, other);
	return *this;
}

CuMat CuMat::Copy() const
{
	CuMat ret(_handle, _rows, _cols, _storageOrder);
	
	if (_dMat)
	{
		cudaError_t status = cudaMemcpy(ret._dMat, _dMat, _rows * _cols * sizeof(Real),
					   cudaMemcpyDeviceToDevice);
		
		if (status != cudaSuccess)
			throw runtime_error("Unable to copy the device memory from this matrix into the copy.");
	}
	
	return ret;
}

void CuMat::CopyToDevice(const Real *hMatrix)
{
	cublasStatus_t status = cublasSetMatrix(_rows, _cols, sizeof(Real),
											hMatrix, _rows, _dMat, _rows);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the host matrix to the device.");
}

void CuMat::CopyToDevice(const CMatrix &hMatrix)
{
	assert(_rows == hMatrix.rows() &&
		   _cols == hMatrix.cols());
	
	CopyToDevice(hMatrix.data());
}

void CuMat::CopyToDevice(const RMatrix &hMatrix)
{
	CMatrix cMat = hMatrix;
	CopyToDevice(cMat);
}

void CuMat::CopyToDeviceAsync(const Real *hMatrix, cudaStream_t stream)
{
	cublasStatus_t status = cublasSetMatrixAsync(_rows, _cols, sizeof(Real),
								hMatrix, _rows, _dMat, _rows, stream);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the host matrix to the device.");
}

void CuMat::CopyToDeviceAsync(const CMatrix &hMatrix, cudaStream_t stream)
{
	assert(_rows == hMatrix.rows() &&
		   _cols == hMatrix.cols());
	
	CopyToDeviceAsync(hMatrix.data(), stream);
}

void CuMat::CopyToDeviceAsync(const RMatrix &hMatrix, cudaStream_t stream)
{
	CMatrix cMat = hMatrix;
	CopyToDeviceAsync(cMat, stream);
}

void CuMat::CopyToHost(Real* hMatrix) const
{
	cublasStatus_t status = cublasGetMatrix(_rows, _cols, sizeof(Real),
											_dMat, _rows, hMatrix, _rows);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the device matrix to the host.");
}

void CuMat::CopyToHost(CMatrix& hMatrix) const
{
	CopyToHost(hMatrix.data());
}

void CuMat::CopyToHost(RMatrix& hMatrix) const
{
	CMatrix cMat(hMatrix.rows(), hMatrix.cols());
	CopyToHost(cMat);
	hMatrix = cMat;
}

void CuMat::CopyToHostAsync(Real* hMatrix, cudaStream_t stream)
{
	cublasStatus_t status = cublasGetMatrixAsync(_rows, _cols, sizeof(Real),
								_dMat, _rows, hMatrix, _rows, stream);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the device matrix to the host.");
}

void CuMat::CopyToHostAsync(CMatrix& hMatrix, cudaStream_t stream)
{
	CopyToHostAsync(hMatrix.data(), stream);
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
