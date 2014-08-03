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

#include "math_defines.h"
#include "cumath_functions.cuh"
#include "cusetup_provider.cuh"

class CuScopedWeakTranspose;
class CuRowwiseOperator;
class CuColwiseOperator;
class CuMat3D;
struct CuMatInfo;

class CuMat
{
    friend class CuRowwiseOperator;
    friend class CuColwiseOperator;
    friend struct CuMatInfo;

public:
	CuMat();
	explicit CuMat(CuContext handle);
	CuMat(CuContext handle, uint32_t rows, uint32_t cols,
	      CuStorageOrder storageOrder = CuColMajor);
	CuMat(CuContext handle, const CMatrix &hMat);
	CuMat(CuContext handle, const RMatrix &hMat);
	CuMat(CuContext handle, const Vector &hVec);
	CuMat(const CuMat &other);
	~CuMat();
	CuMat &operator=(CuMat other);
	CuMat Copy() const;
	
	bool Empty() const;
	bool SingleOwner() const;

	CuContext Handle() const { return _handle; }
	CuContext &Handle() { return _handle; }
	uint32_t Rows() const { return _rows; }
	uint32_t Cols() const { return _cols; }
	uint32_t Size() const { return _rows * _cols; }
	Real *Buff() { return _dMat; }
	const Real *Buff() const { return _dMat; }
	CuStorageOrder Order() const { return _storageOrder; }

	friend CuMat operator+(const CuMat &a, const CuMat &b);
	friend CuMat operator-(const CuMat &a, const CuMat &b);

	friend CuMat operator*(const CuMat &a, const CuMat &b);
	friend CuMat operator*(const CuScopedWeakTranspose &a, const CuMat &b);
	friend CuMat operator*(const CuMat &a, const CuScopedWeakTranspose &b);
	friend CuMat operator*(const CuScopedWeakTranspose &a, const CuScopedWeakTranspose &b);

	friend CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b, cublasHandle_t cublasHandle = 0);
	friend CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b, cublasHandle_t cublasHandle = 0);
	friend CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle = 0);
	friend CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle = 0);

	friend void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle = 0);
	friend void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle = 0);
	friend void ScaledMultiply(Real mulScale, const CuMat &a, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle = 0);
	friend void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle = 0);

	friend CuMat MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols, const CuMat &b);
    friend void MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols, const CuMat &b, CuMat &dest);

	friend CuMat &operator+=(CuMat &a, const CuMat &b);
	friend CuMat &operator-=(CuMat &a, const CuMat &b);

	CuMat &operator=(Real val);
	CuMat &operator*=(Real val);
	CuMat &operator/=(Real val);

	void SetConstant(Real val);

	void CoeffMultiply(Real val);
	void CoeffMultiply(Real val, CuMat &dest) const;
	void CoeffMultiply(const CuMat &b);
	void CoeffMultiply(const CuMat &b, CuMat &dest) const;

	void AddScaled(Real scaleThis, const CuMat &b, Real scaleB);
	void AddScaled(Real scaleThis, const CuMat &b, Real scaleB, CuMat &dest) const;

	friend void AddScaled(const CuMat &a, Real scaleA, const CuMat &b, Real scaleB, CuMat &dest);

	void CopyToDevice(const Real *hMatrix);
	void CopyToDevice(const CMatrix &hMatrix);
	void CopyToDevice(const RMatrix &hMatrix);
	void CopyToDevice(const Vector &hVector);
	void CopyToDeviceAsync(const Real *hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const CMatrix &hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const RMatrix &hMatrix, cudaStream_t stream);
	
	void CopyToHost(Real *hMatrix) const;
	void CopyToHost(CMatrix &hMatrix) const;
	void CopyToHost(RMatrix &hMatrix) const;
	void CopyToHost(Vector &hVector) const;
	void CopyToHostAsync(Real *hMatrix, cudaStream_t stream);
	void CopyToHostAsync(CMatrix &hMatrix, cudaStream_t stream);

	template<typename UnaryFn>
	void UnaryExpr(UnaryFn fn);
	template<bool Add, typename UnaryFn>
	void UnaryExpr(CuMat &dest, UnaryFn fn) const;

	template<typename BinaryFn>
	void BinaryExpr(const CuMat &b, BinaryFn fn);
	template<bool Add, typename BinaryFn>
	void BinaryExpr(const CuMat &b, CuMat &dest, BinaryFn fn) const;

	template<typename TrinaryFn>
	void TrinaryExpr(const CuMat &b, const CuMat &c, TrinaryFn fn);
	template<bool Add, typename TrinaryFn>
	void TrinaryExpr(const CuMat &b, const CuMat &c, CuMat &dest, TrinaryFn fn) const;

	void Resize(uint32_t rows, uint32_t cols);
	void ResizeLike(const CuMat &like);
	void Reshape(uint32_t rows, uint32_t cols);

	CuMat Transpose() const;
	CuMat HardTranspose() const;
	CuScopedWeakTranspose WeakTranspose() const;

	CuRowwiseOperator Rowwise() const;
	CuColwiseOperator Colwise() const;

	Real Sum() const;

	void SetHandle(const CuContext &handle) { _handle = handle; }

	void SetStream(cudaStream_t stream) { _handle.SetStream(stream); }

	void SetSharedModify(bool val) { _sharedMod = val; }

	friend void swap(CuMat &a, CuMat &b);
	
private:
	void SetDevice() const;
	void PrepareForWrite(bool alloc);
	void AllocateMatrix();
	void FreeMatrix();
	void AssertSameDims(const CuMat &other) const;

	CuMatInfo ToInfo() const;

	Real *_dMat;
	uint32_t *_refCt;
	uint32_t _rows, _cols, _buffSize;
	CuStorageOrder _storageOrder;
	CuContext _handle;
	bool _sharedMod;
};

struct CuMatInfo
{
    friend class CuMat;

    Real *_dMat;
    uint32_t _rows, _cols;
    CuStorageOrder _storageOrder;

    CuMatInfo() : _dMat(NULL), _rows(0), _cols(0),
                  _storageOrder(CuColMajor) { }

    CuMatInfo(const CuMat &m)
        : _dMat(m._dMat), _rows(m._rows), _cols(m._cols),
          _storageOrder(m._storageOrder) { }
};

class CuScopedWeakTranspose
{
public:
    CuScopedWeakTranspose(const CuMat &mat);

    const CuMat &Mat;
};

template<typename Inc>
class CuMatAgg_t
{
public:
    CuMatAgg_t(const CuMat &mat)
        : Mat(mat) { }

    const CuMat &Mat;

    CuMat Sum() const;
    template<typename ElemFn>
    CuMat Sum(ElemFn fn) const;

    void Sum(CuMat &dest, cublasHandle_t cublasHandle = 0) const;
    template<typename ElemFn>
    void Sum(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle = 0) const;

    CuMat Max() const;
    template<typename ElemFn>
    CuMat Max(ElemFn fn) const;

    void Max(CuMat &dest, cublasHandle_t cublasHandle = 0) const;
    template<typename ElemFn>
    void Max(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle = 0) const;

    CuMat MaxIdx() const;
    template<typename ElemFn>
    CuMat MaxIdx(ElemFn fn) const;

    void MaxIdx(CuMat &dest, cublasHandle_t cublasHandle = 0) const;
    template<typename ElemFn>
    void MaxIdx(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle = 0) const;

    CuMat Min() const;
    template<typename ElemFn>
    CuMat Min(ElemFn fn) const;

    void Min(CuMat &dest, cublasHandle_t cublasHandle = 0) const;
    template<typename ElemFn>
    void Min(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle = 0) const;

    CuMat MinIdx() const;
    template<typename ElemFn>
    CuMat MinIdx(ElemFn fn) const;

    void MinIdx(CuMat &dest, cublasHandle_t cublasHandle = 0) const;
    template<typename ElemFn>
    void MinIdx(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle = 0) const;

    template<bool Vals, typename Aggregator, typename ElemFn>
    CuMat Agg(Aggregator agg, ElemFn fn) const;

    template<bool Vals, typename Aggregator, typename ElemFn>
    void Agg(CuMat &dest, Aggregator agg, ElemFn fn, cublasHandle_t cublasHandle = 0) const;
};

template<uint32_t RowInc, uint32_t ColInc>
class Incrementer
{
public:
    static const bool IsHorizontal = ColInc >= 1;
    static const bool IsVertical = RowInc >= 1;

    __device__ __host__ void operator()(uint32_t &row, uint32_t &col, uint32_t stride = 1) const
    {
        row += RowInc * stride;
        col += ColInc * stride;
    }

    static uint32_t XDim(uint32_t cols)
    {
        if (IsHorizontal)
            return 1;
        else
            return cols;
    }
    static uint32_t YDim(uint32_t rows)
    {
        if (IsVertical)
            return 1;
        else
            return rows;
    }
};

class CuRowwiseOperator
    : public CuMatAgg_t<Incrementer<0, 1> >
{
public:
    CuRowwiseOperator(const CuMat &mat)
        : CuMatAgg_t(mat) { }

};

class CuColwiseOperator
    : public CuMatAgg_t<Incrementer<1, 0> >
{
public:
    CuColwiseOperator(const CuMat &mat)
        : CuMatAgg_t(mat) { }
};

CuMat operator*(const CuMat &m, Real scale);
CuMat operator*(Real scale, const CuMat &m);
CuMat operator/(const CuMat &m, Real scale);
CuMat operator+(const CuMat &m, Real val);
CuMat operator+(Real val, const CuMat &m);
CuMat operator-(const CuMat &m, Real val);
CuMat operator-(Real val, const CuMat &m);

CuMat operator*(const CuScopedWeakTranspose &a, const CuScopedWeakTranspose &b);

CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b, cublasHandle_t cublasHandle);
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b, cublasHandle_t cublasHandle);
CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle);
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB, cublasHandle_t cublasHandle);

void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle);
void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuMat &b, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle);
void ScaledMultiply(Real mulScale, const CuMat &a, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle);
void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA,
		const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest, cublasHandle_t cublasHandle);

CuMat MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols, const CuMat &b);
void MultiplyTrans3D(const CuMat &a, uint32_t rows, uint32_t cols,
        const CuMat &b, CuMat &dest);



#include "cumat_kernels.cuh"
#include "cumat_temp_agg.cuh"
#include "cumat3d.cuh"
