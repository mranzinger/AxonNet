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
#include "cumath_functions.cuh"

class CuScopedWeakTranspose;
class CuRowwiseOperator;
class CuColwiseOperator;
struct CuMatInfo;

class CuMat
{
    friend class CuRowwiseOperator;
    friend class CuColwiseOperator;
    friend struct CuMatInfo;

public:
	CuMat();
	explicit CuMat(cublasHandle_t handle);
	CuMat(cublasHandle_t handle, uint32_t rows, uint32_t cols,
	      CuStorageOrder storageOrder = CuColMajor);
	CuMat(const CuMat &other);
	~CuMat();
	CuMat &operator=(CuMat other);
	CuMat Copy() const;
	
	bool Empty() const;
	bool SingleOwner() const;

	cublasHandle_t Handle() const { return _handle; }
	uint32_t Rows() const { return _rows; }
	uint32_t Cols() const { return _cols; }
	const Real *Buff() const { return _dMat; }

	friend CuMat operator+(const CuMat &a, const CuMat &b);
	friend CuMat operator-(const CuMat &a, const CuMat &b);

	friend CuMat operator*(const CuMat &a, const CuMat &b);
	friend CuMat operator*(const CuScopedWeakTranspose &a, const CuMat &b);
	friend CuMat operator*(const CuMat &a, const CuScopedWeakTranspose &b);
	friend CuMat operator*(const CuScopedWeakTranspose &a, const CuScopedWeakTranspose &b);

	friend CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b);
	friend CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b);
	friend CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB);
	friend CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB);

	friend void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest);
	friend void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuMat &b, Real scaleDest, CuMat &dest);
	friend void ScaledMultiply(Real mulScale, const CuMat &a, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest);
	friend void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest);

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
	void CopyToDeviceAsync(const Real *hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const CMatrix &hMatrix, cudaStream_t stream);
	void CopyToDeviceAsync(const RMatrix &hMatrix, cudaStream_t stream);
	
	void CopyToHost(Real *hMatrix) const;
	void CopyToHost(CMatrix &hMatrix) const;
	void CopyToHost(RMatrix &hMatrix) const;
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

	friend void swap(CuMat &a, CuMat &b);
	
private:
	void PrepareForWrite(bool alloc);
	void AllocateMatrix();
	void FreeMatrix();
	void AssertSameDims(const CuMat &other) const;

	CuMatInfo ToInfo() const;

	Real *_dMat;
	uint32_t *_refCt;
	uint32_t _rows, _cols;
	CuStorageOrder _storageOrder;
	cublasHandle_t _handle;
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

class CuRowwiseOperator
{
public:
    CuRowwiseOperator(const CuMat &mat);

    const CuMat &Mat;

    CuMat Sum() const;
    template<typename ElemFn>
    CuMat Sum(ElemFn fn) const;

    CuMat Max() const;
    template<typename ElemFn>
    CuMat Max(ElemFn fn) const;

    CuMat Min() const;
    template<typename ElemFn>
    CuMat Min(ElemFn fn) const;

    template<typename Aggregator, typename ElemFn>
    CuMat Agg(Aggregator agg, ElemFn fn) const;
};

class CuColwiseOperator
{
public:
    CuColwiseOperator(const CuMat &mat);

    const CuMat &Mat;

    CuMat Sum() const;
    template<typename ElemFn>
    CuMat Sum(ElemFn fn) const;

    CuMat Max() const;
    template<typename ElemFn>
    CuMat Max(ElemFn fn) const;

    CuMat Min() const;
    template<typename ElemFn>
    CuMat Min(ElemFn fn) const;

    template<typename Aggregator, typename ElemFn>
    CuMat Agg(Aggregator agg, ElemFn fn) const;
};

CuMat operator*(const CuMat &m, Real scale);
CuMat operator*(Real scale, const CuMat &m);
CuMat operator/(const CuMat &m, Real scale);
CuMat operator+(const CuMat &m, Real val);
CuMat operator+(Real val, const CuMat &m);
CuMat operator-(const CuMat &m, Real val);
CuMat operator-(Real val, const CuMat &m);

CuMat operator*(const CuScopedWeakTranspose &a, const CuScopedWeakTranspose &b);

CuMat ScaledMultiply(Real scale, const CuMat &a, const CuMat &b);
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuMat &b);
CuMat ScaledMultiply(Real scale, const CuMat &a, const CuScopedWeakTranspose &tB);
CuMat ScaledMultiply(Real scale, const CuScopedWeakTranspose &tA, const CuScopedWeakTranspose &tB);

void ScaledMultiply(Real mulScale, const CuMat &a, const CuMat &b, Real scaleDest, CuMat &dest);
void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA, const CuMat &b, Real scaleDest, CuMat &dest);
void ScaledMultiply(Real mulScale, const CuMat &a, const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest);
void ScaledMultiply(Real mulScale, const CuScopedWeakTranspose &tA,
		const CuScopedWeakTranspose &tB, Real scaleDest, CuMat &dest);

#include "cumat_kernels.cuh"
#include "cumat_temp_agg.cuh"
