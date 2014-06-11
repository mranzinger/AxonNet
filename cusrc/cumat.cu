#include "cumat.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <assert.h>

using namespace std;

CuMat::CuMat()
	: _dMat(NULL), _rows(0), _cols(0), _storageOrder(CuColMajor), _sharedMod(false)
{
	_refCt = new uint32_t(1);
}

CuMat::CuMat(CuContext handle)
    : _handle(handle), _dMat(NULL), _rows(0), _cols(0), _storageOrder(CuColMajor), _sharedMod(false)
{
    _refCt = new uint32_t(1);
}

CuMat::CuMat(CuContext handle,
		     uint32_t rows, uint32_t cols,
		     CuStorageOrder storageOrder)
	: _handle(handle), _dMat(NULL), _rows(rows), _cols(cols),
	  _storageOrder(storageOrder), _sharedMod(false)
{
	_refCt = new uint32_t(1);
	
	AllocateMatrix();
}

CuMat::CuMat(CuContext handle, const CMatrix& hMat)
    : _handle(handle), _rows(hMat.rows()), _cols(hMat.cols()),
      _storageOrder(CuColMajor), _sharedMod(false)
{
    _refCt = new uint32_t(1);

    AllocateMatrix();

    CopyToDevice(hMat);
}

CuMat::CuMat(CuContext handle, const RMatrix& hMat)
    : _handle(handle), _rows(hMat.rows()), _cols(hMat.cols()),
      _storageOrder(CuRowMajor), _sharedMod(false)
{
    _refCt = new uint32_t(1);

    AllocateMatrix();

    CopyToDevice(hMat);
}

CuMat::CuMat(CuContext handle, const Vector& hVec)
    : _handle(handle), _rows(hVec.rows()), _cols(hVec.cols()),
      _storageOrder(CuRowMajor), _sharedMod(false)
{
    _refCt = new uint32_t(1);

    AllocateMatrix();

    CopyToDevice(hVec);
}

CuMat::CuMat(const CuMat &other)
	: _handle(other._handle), _dMat(other._dMat), _rows(other._rows), _cols(other._cols),
	  _refCt(other._refCt), _storageOrder(other._storageOrder), _sharedMod(other._sharedMod)
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
	CuMat ret(_handle, _rows, _cols);
	
	if (_dMat)
	{
		SetDevice();

		cudaError_t status = cudaMemcpy(ret._dMat, _dMat, _rows * _cols * sizeof(Real),
					   cudaMemcpyDeviceToDevice);
		
		if (status != cudaSuccess)
			throw runtime_error("Unable to copy the device memory from this matrix into the copy.");
	}
	
	return ret;
}

void CuMat::CopyToDevice(const Real *hMatrix)
{
	SetDevice();

	cublasStatus_t status = cublasSetMatrix(_rows, _cols, sizeof(Real),
											hMatrix, _rows, _dMat, _rows);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the host matrix to the device.");
}

void CuMat::CopyToDevice(const CMatrix &hMatrix)
{
	Resize(hMatrix.rows(), hMatrix.cols());

	if (_storageOrder == CuColMajor)
	{
	    CopyToDevice(hMatrix.data());
	}
	else
	{
	    CuMat dColMat(_handle, hMatrix);

	    // Copy the row-major matrix into this column
        // major matrix
	    BinaryExpr(dColMat, CuTakeRight());
	}
}

void CuMat::CopyToDevice(const RMatrix &hMatrix)
{
    Resize(hMatrix.rows(), hMatrix.cols());

    if (_storageOrder == CuRowMajor)
    {
        CopyToDevice(hMatrix.data());
    }
    else
    {
        CuMat dRowMat(_handle, hMatrix);

        // Copy the row-major matrix into this column
        // major matrix
        BinaryExpr(dRowMat, CuTakeRight());
    }
}

void CuMat::CopyToDevice(const Vector &hVector)
{
    Resize(hVector.size(), 1);

    CopyToDevice(hVector.data());
}

void CuMat::CopyToDeviceAsync(const Real *hMatrix, cudaStream_t stream)
{
	SetDevice();

	cublasStatus_t status = cublasSetMatrixAsync(_rows, _cols, sizeof(Real),
								hMatrix, _rows, _dMat, _rows, stream);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the host matrix to the device.");
}

void CuMat::CopyToDeviceAsync(const CMatrix &hMatrix, cudaStream_t stream)
{
    Resize(hMatrix.rows(), hMatrix.cols());

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
	SetDevice();

	cublasStatus_t status = cublasGetMatrix(_rows, _cols, sizeof(Real),
											_dMat, _rows, hMatrix, _rows);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the device matrix to the host.");
}

void CuMat::CopyToHost(CMatrix& hMatrix) const
{
	if (_rows != hMatrix.rows() ||
	    _cols != hMatrix.cols())
	{
		hMatrix.resize(_rows, _cols);
	}

	CopyToHost(hMatrix.data());
}

void CuMat::CopyToHost(Vector& hVector) const
{
    assert(_cols == 1);

    if (_rows != hVector.size())
    {
        hVector.resize(_rows);
    }

    CopyToHost(hVector.data());
}

void CuMat::CopyToHost(RMatrix& hMatrix) const
{
	CMatrix cMat(hMatrix.rows(), hMatrix.cols());
	CopyToHost(cMat);
	hMatrix = cMat;
}

void CuMat::CopyToHostAsync(Real* hMatrix, cudaStream_t stream)
{
	SetDevice();

	cublasStatus_t status = cublasGetMatrixAsync(_rows, _cols, sizeof(Real),
								_dMat, _rows, hMatrix, _rows, stream);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("Unable to copy the device matrix to the host.");
}

void CuMat::CopyToHostAsync(CMatrix& hMatrix, cudaStream_t stream)
{
	CopyToHostAsync(hMatrix.data(), stream);
}

CuMat& CuMat::operator =(Real val)
{
    SetConstant(val);
    return *this;
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



void CuMat::SetConstant(Real val)
{
    PrepareForWrite(false);

    UnaryExpr(CuConstant(val));
}

void CuMat::Resize(uint32_t rows, uint32_t cols)
{
	// Test for a no-op
	if ((SingleOwner() || _sharedMod) && _rows == rows && _cols == cols)
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
	if (*_refCt == 1 || _sharedMod)
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

	SetDevice();

	cudaError_t cudaStat = cudaMalloc(&_dMat, _rows * _cols * sizeof(Real));
	if (cudaStat != cudaSuccess)
		throw runtime_error("Unable to allocate the specified matrix");
}

void CuMat::FreeMatrix()
{
	SetDevice();

	// Free the device memory
	cudaFree(_dMat);
}

CuMat CuMat::Transpose() const
{
    CuMat ret(*this);

    swap(ret._rows, ret._cols);
    ret._storageOrder = InverseOrder(_storageOrder);

    return ret;
}

CuMat CuMat::HardTranspose() const
{
    CuMat ret(_handle, _rows, _cols, InverseOrder(_storageOrder));

    UnaryExpr<false>(ret, CuIdentity());

    swap(ret._rows, ret._cols);
    ret._storageOrder = _storageOrder;

    return ret;
}

CuScopedWeakTranspose CuMat::WeakTranspose() const
{
    return CuScopedWeakTranspose(*this);
}

void CuMat::AssertSameDims(const CuMat& other) const
{
	if (_rows != other._rows)
		throw runtime_error("The specified matrix doesn't have the same number of rows as this one.");
	if (_cols != other._cols)
		throw runtime_error("The specified matrix doesn't have the same number of columns as this one.");
}

void swap(CuMat &a, CuMat &b)
{
	swap(a._handle, b._handle);
	swap(a._dMat, b._dMat);
	swap(a._rows, b._rows);
	swap(a._cols, b._cols);
	swap(a._refCt, b._refCt);
	swap(a._sharedMod, b._sharedMod);
}

CuScopedWeakTranspose::CuScopedWeakTranspose(const CuMat& mat)
	: Mat(mat)
{
}

CuRowwiseOperator CuMat::Rowwise() const
{
	return CuRowwiseOperator(*this);
}

CuColwiseOperator CuMat::Colwise() const
{
	return CuColwiseOperator(*this);
}

void CuMat::SetDevice() const
{
	cudaSetDevice(_handle.Device);
}

CuMatInfo CuMat::ToInfo() const
{
    return CuMatInfo(*this);
}

Real CuMat::Sum() const
{
	if (Empty())
		return 0.0f;

	Real ret;
	cublasStatus_t status = cublasSasum_v2(_handle.CublasHandle,
											Size(),
											_dMat,
											1,
											&ret);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("An error occurred while attempting to compute the sum of this matrix.");

	return ret;
}
