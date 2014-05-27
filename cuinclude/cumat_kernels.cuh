/*
 * cumat_kernels.h
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#include "cumat.cuh"
#include "cumath_functions.cuh"

template<typename UnaryFn>
void CuMat::UnaryExpr(UnaryFn fn)
{
	UnaryExpr<false>(*this, fn);
}

template<bool Add, typename UnaryFn>
void CuMat::UnaryExpr(CuMat& dest, UnaryFn fn) const
{
	dest.ResizeLike(*this);

	ApplyUnaryFn<Add>(_dMat, dest._dMat, _rows, _cols,
					  _storageOrder, dest._storageOrder, fn);
}

template<typename BinaryFn>
void CuMat::BinaryExpr(const CuMat& b, BinaryFn fn)
{
	BinaryExpr<false>(b, *this, fn);
}

template<bool Add, typename BinaryFn>
void CuMat::BinaryExpr(const CuMat& b, CuMat& dest, BinaryFn fn) const
{
	// Make sure the other matrix has the same dimensions
	AssertSameDims(b);

	dest.ResizeLike(*this);

	ApplyBinaryFn<Add>(_dMat, b._dMat, dest._dMat, _rows, _cols,
					  _storageOrder, b._storageOrder, dest._storageOrder,
					  fn);
}

template<typename TrinaryFn>
void CuMat::TrinaryExpr(const CuMat& b, const CuMat& c, TrinaryFn fn)
{
	TrinaryExpr<false>(b, c, *this, fn);
}

template<bool Add, typename TrinaryFn>
void CuMat::TrinaryExpr(const CuMat& b, const CuMat& c, CuMat& dest,
		TrinaryFn fn) const
{
	AssertSameDims(b);
	AssertSameDims(c);

	dest.ResizeLike(*this);

	ApplyTrinaryFn<Add>(_dMat, b._dMat, c._dMat, dest._dMat, _rows, _cols,
						_storageOrder, b._storageOrder, c._storageOrder, dest._storageOrder,
						fn);
}
