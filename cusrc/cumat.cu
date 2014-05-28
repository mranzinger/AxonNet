#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <assert.h>

using namespace std;

CuMat::CuMat()
	: _handle(0), _dMat(NULL), _rows(0), _cols(0), _storageOrder(CuColMajor)
{
	_refCt = new uint32_t(1);
}

CuMat::CuMat(cublasHandle_t handle,
		     uint32_t rows, uint32_t cols,
		     CuStorageOrder order)
	: _handle(handle), _dMat(NULL), _rows(rows), _cols(cols), _storageOrder(order)
{
	_refCt = new uint32_t(1);
	
	AllocateMatrix();
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
		FreeMatrix();
	}
}

bool CuMat::Empty() const
{
	return !_dMat || !_rows || !_cols;
}

bool CuMat::SingleOwner() const
{
	return *_refCt == 1;
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

CuMat operator+(const CuMat &a, const CuMat &b)
{
	CuMat ret;
	a.BinaryExpr<false>(b, ret, CuPlus());
	return ret;
}
CuMat operator-(const CuMat &a, const CuMat &b)
{
	CuMat ret;
	a.BinaryExpr<false>(b, ret, CuMinus());
	return ret;
}
CuMat operator*(const CuMat &a, const CuMat &b)
{
	static const float s_default = 1.0f;

	// Make sure the matrices are valid
	assert(a._cols == b._rows);
	assert(!a.Empty() && !b.Empty());
	assert(a._handle == b._handle);

	CuMat ret(a._handle, a._rows, b._cols);

	cublasStatus_t status =
			cublasSgemm_v2(a._handle, a.GetTransOrder(), b.GetTransOrder(),
							a._rows, b._cols, a._cols,
							&s_default, a._dMat, a._rows,
							b._dMat, b._rows,
							NULL,
							NULL,
							0);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("The matrix multiplication failed.");

	return ret;
}

CuMat &operator+=(CuMat &a, const CuMat &b)
{
	a.BinaryExpr(b, CuPlus());
	return a;
}
CuMat &operator-=(CuMat &a, const CuMat &b)
{
	a.BinaryExpr(b, CuMinus());
	return a;
}

void CuMat::CoeffMultiply(Real val)
{
	CoeffMultiply(val, *this);
}

void CuMat::CoeffMultiply(Real val, CuMat& dest) const
{
	UnaryExpr<false>(dest, CuUnaryScale(val));
}

void CuMat::CoeffMultiply(const CuMat& b)
{
	CoeffMultiply(b, *this);
}

void CuMat::CoeffMultiply(const CuMat& b, CuMat& dest) const
{
	BinaryExpr<false>(b, dest, CuMultiply());
}

void CuMat::AddScaled(Real scaleThis, const CuMat& b, Real scaleB)
{
	AddScaled(scaleThis, b, scaleB, *this);
}

void CuMat::AddScaled(Real scaleThis, const CuMat& b, Real scaleB,
		CuMat& dest) const
{
	AssertSameDims(b);

	BinaryExpr<false>(b, dest, CuAddScaledBinary(scaleThis, scaleB));
}

void CuMat::Resize(uint32_t rows, uint32_t cols)
{
	// Test for a no-op
	if (SingleOwner() && _rows == rows && _cols == cols)
		return;

	// Ensure exclusive ownership of the matrix before
	// modifying it
	PrepareForWrite(false);

	_rows = rows;
	_cols = cols;

	// Free the old buffer if it is valid
	FreeMatrix();

	// Allocate the new matrix of the specified size
	AllocateMatrix();
}

void CuMat::ResizeLike(const CuMat& like)
{
	Resize(like._rows, like._cols);
}

void CuMat::Reshape(uint32_t rows, uint32_t cols)
{
	throw runtime_error("Not implemented.");
}

void CuMat::PrepareForWrite(bool alloc)
{
	// This is a copy on modify paradigm,
	// so if this instance is a sole owner of the data,
	// then nothing needs to be done
	if (*_refCt == 1)
		return;

	_refCt = new uint32_t(1);

	if (alloc)
		AllocateMatrix();
	else
		_dMat = NULL;
}

void CuMat::AllocateMatrix()
{
	_dMat = NULL;

	if (_rows == 0 || _cols == 0)
		return;

	cudaError_t cudaStat = cudaMalloc(&_dMat, _rows * _cols * sizeof(Real));
	if (cudaStat != cudaSuccess)
		throw runtime_error("Unable to allocate the specified matrix");
}

void CuMat::FreeMatrix()
{
	// Free the device memory
	cudaFree(_dMat);
}



void CuMat::AssertSameDims(const CuMat& other) const
{
	if (_rows != other._rows)
		throw runtime_error("The specified matrix doesn't have the same number of rows as this one.");
	if (_cols != other._cols)
		throw runtime_error("The specified matrix doesn't have the same number of columns as this one.");
}

CuStorageOrder CuMat::InverseOrder(CuStorageOrder order)
{
	switch (order)
	{
	case CuColMajor:
		return CuRowMajor;
	case CuRowMajor:
		return CuColMajor;
	default:
		throw runtime_error("Invalid storage order");
	}
}

cublasOperation_t CuMat::GetTransOrder() const
{
	switch (_storageOrder)
	{
	case CuColMajor:
		return CUBLAS_OP_N;
	case CuRowMajor:
		return CUBLAS_OP_T;
	default:
		throw runtime_error("Invalid storage order");
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

CuScopedWeakTranspose::CuScopedWeakTranspose(CuMat& mat)
	: _mat(mat)
{
	Invert();
}

CuScopedWeakTranspose::~CuScopedWeakTranspose()
{
	// Undo the inversion
	Invert();
}

void CuScopedWeakTranspose::Invert()
{
	swap(_mat._rows, _mat._cols);

	_mat._storageOrder = CuMat::InverseOrder(_mat._storageOrder);
}
