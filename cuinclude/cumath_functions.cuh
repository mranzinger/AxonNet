/*
 * cumath_functions.cuh
 *
 *  Created on: May 26, 2014
 *      Author: mike
 */


#pragma once

#include <stdexcept>
#include <stdint.h>
#include <assert.h>

#define __CUDACC__
#include <cuda_runtime_api.h>

#include "cumath_unary_functions.cuh"
#include "cumath_binary_functions.cuh"

inline uint32_t round_up(uint32_t val, uint32_t base)
{
	return (val + base - 1) / base;
}

inline dim3 round_up(uint32_t x, uint32_t y, uint32_t z, uint32_t base)
{
	return dim3(round_up(x, base),
			    round_up(y, base),
			    round_up(z, base));
}

template<uint32_t base>
dim3 round_up(uint32_t x, uint32_t y = 1, uint32_t z = 1)
{
	return dim3((x + base - 1) / base,
				(y + base - 1) / base,
				(z + base - 1) / base);
}

enum CuStorageOrder
{
	CuRowMajor,
	CuColMajor
};

template<CuStorageOrder order>
__device__ inline unsigned int ElementIdx(unsigned int row, unsigned int col,
						       unsigned int rows, unsigned int cols)
{
	if (order == CuRowMajor)
		return row * cols + col;
	else
		return col * rows + row;
}

template<typename UnaryFn,
		 CuStorageOrder orderSrc,
		 CuStorageOrder orderDest,
		 bool Add>
__global__ void DApplyUnaryFn(const Real *pVecSrc, Real *pVecTarget,
						     unsigned int rows, unsigned int cols, UnaryFn fn)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= rows || col >= cols)
		return;

	const unsigned int srcIdx = ElementIdx<orderSrc>(row, col, rows, cols),
					   destIdx = ElementIdx<orderDest>(row, col, rows, cols);

	const Real srcVal = fn(pVecSrc[srcIdx]);
	Real &destVal = pVecTarget[destIdx];

	if (Add)
		destVal += srcVal;
	else
		destVal = srcVal;
}



template<typename BinaryFn,
		 CuStorageOrder orderA,
		 CuStorageOrder orderB,
		 CuStorageOrder orderDest,
		 bool Add>
__global__ void DApplyBinaryFn(const Real *pVecA, const Real *pVecB,
							  Real *pVecTarget,
							  unsigned int rows, unsigned int cols, BinaryFn fn)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= rows || col >= cols)
		return;

	const unsigned int idxA = ElementIdx<orderA>(row, col, rows, cols),
					   idxB = ElementIdx<orderB>(row, col, rows, cols),
					   idxDest = ElementIdx<orderDest>(row, col, rows, cols);

	const Real srcVal = fn(pVecA[idxA], pVecB[idxB]);
	Real &destVal = pVecTarget[idxDest];

	if (Add)
		destVal += srcVal;
	else
		destVal = srcVal;
}

template<typename TrinaryFn,
		 CuStorageOrder orderA,
		 CuStorageOrder orderB,
		 CuStorageOrder orderC,
		 CuStorageOrder orderDest,
		 bool Add>
__global__ void DApplyTrinaryFn(const Real *pVecA, const Real *pVecB, const Real *pVecC,
							   Real *pVecTarget,
							   unsigned int rows, unsigned int cols,
							   TrinaryFn fn)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= rows || col >= cols)
		return;

	const unsigned int idxA = ElementIdx<orderA>(row, col, rows, cols),
					   idxB = ElementIdx<orderB>(row, col, rows, cols),
					   idxC = ElementIdx<orderC>(row, col, rows, cols),
					   idxDest = ElementIdx<orderDest>(row, col, rows, cols);

	const Real srcVal = fn(pVecA[idxA], pVecB[idxB], pVecC[idxC]);
	Real &destVal = pVecTarget[idxDest];

	if (Add)
		destVal += srcVal;
	else
		destVal = srcVal;
}

template<bool Add, typename UnaryFn>
void ApplyUnaryFn(const Real *pVecSrc, Real *pVecTarget,
				  unsigned int rows, unsigned int cols,
				  CuStorageOrder orderA,
				  CuStorageOrder orderDest, UnaryFn fn)
{
	dim3 blockSize = round_up<32>(cols, rows);

#define CALL_UNARY(orderA, orderDest) \
		DApplyUnaryFn<UnaryFn, orderA, orderDest, Add> \
					 <<<blockSize, dim3(min(32, cols), min(32, rows))>>> \
				     (pVecSrc, pVecTarget, rows, cols, fn)

	if (orderA == CuColMajor)
	{
		if (orderDest == CuColMajor)
		{
			CALL_UNARY(CuColMajor, CuColMajor);
		}
		else
		{
			CALL_UNARY(CuColMajor, CuRowMajor);
		}
	}
	else
	{
		if (orderDest == CuColMajor)
		{
			CALL_UNARY(CuRowMajor, CuColMajor);
		}
		else
		{
			CALL_UNARY(CuRowMajor, CuRowMajor);
		}
	}

#undef CALL_UNARY
}

template<bool Add, typename BinaryFn>
void ApplyBinaryFn(const Real *pVecA, const Real *pVecB,
				   Real *pVecTarget,
				   unsigned int rows, unsigned int cols,
				   CuStorageOrder orderA,
				   CuStorageOrder orderB,
				   CuStorageOrder orderDest, BinaryFn fn)
{
	dim3 blockSize = round_up<32>(cols, rows);

#define CALL_BINARY(orderA, orderB, orderDest) \
	DApplyBinaryFn<BinaryFn, orderA, orderB, orderDest, Add> \
				   <<<blockSize, dim3(min(32, cols), min(32, rows))>>> \
				   (pVecA, pVecB, pVecTarget, rows, cols, fn);

	if (orderA == CuColMajor)
	{
		if (orderB == CuColMajor)
		{
			if (orderDest == CuColMajor)
			{
				CALL_BINARY(CuColMajor, CuColMajor, CuColMajor);
			}
			else
			{
				CALL_BINARY(CuColMajor, CuColMajor, CuRowMajor);
			}
		}
		else
		{
			if (orderDest == CuColMajor)
			{
				CALL_BINARY(CuColMajor, CuRowMajor, CuColMajor);
			}
			else
			{
				CALL_BINARY(CuColMajor, CuRowMajor, CuRowMajor);
			}
		}
	}
	else
	{
		if (orderB == CuColMajor)
		{
			if (orderDest == CuColMajor)
			{
				CALL_BINARY(CuRowMajor, CuColMajor, CuColMajor);
			}
			else
			{
				CALL_BINARY(CuRowMajor, CuColMajor, CuRowMajor);
			}
		}
		else
		{
			if (orderDest == CuColMajor)
			{
				CALL_BINARY(CuRowMajor, CuRowMajor, CuColMajor);
			}
			else
			{
				CALL_BINARY(CuRowMajor, CuRowMajor, CuRowMajor);
			}
		}
	}

#undef CALL_BINARY
}

template<bool Add, typename TrinaryFn>
void ApplyTrinaryFn(const Real *pVecA, const Real *pVecB, const Real *pVecC,
				    Real *pVecTarget,
				    unsigned int rows, unsigned int cols,
				   CuStorageOrder orderA,
				   CuStorageOrder orderB,
				   CuStorageOrder orderC,
				   CuStorageOrder orderDest, TrinaryFn fn)
{
	dim3 blockSize = round_up<32>(cols, rows);

#define CALL_TRINARY(orderA, orderB, orderC, orderDest) \
	DApplyTrinaryFn<TrinaryFn, orderA, orderB, orderC, orderDest, Add> \
					<<<blockSize, dim3(min(32, cols), min(32, rows))>>> \
					(pVecA, pVecB, pVecC, pVecTarget, rows, cols, fn);

	if (orderA == CuColMajor)
	{
		if (orderB == CuColMajor)
		{
			if (orderC == CuColMajor)
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuColMajor, CuColMajor, CuColMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuColMajor, CuColMajor, CuColMajor, CuRowMajor);
				}
			}
			else
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuColMajor, CuColMajor, CuRowMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuColMajor, CuColMajor, CuRowMajor, CuRowMajor);
				}
			}
		}
		else
		{
			if (orderC == CuColMajor)
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuColMajor, CuRowMajor, CuColMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuColMajor, CuRowMajor, CuColMajor, CuRowMajor);
				}
			}
			else
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuColMajor, CuRowMajor, CuRowMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuColMajor, CuRowMajor, CuRowMajor, CuRowMajor);
				}
			}
		}
	}
	else
	{
		if (orderB == CuColMajor)
		{
			if (orderC == CuColMajor)
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuRowMajor, CuColMajor, CuColMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuRowMajor, CuColMajor, CuColMajor, CuRowMajor);
				}
			}
			else
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuRowMajor, CuColMajor, CuRowMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuRowMajor, CuColMajor, CuRowMajor, CuRowMajor);
				}
			}
		}
		else
		{
			if (orderC == CuColMajor)
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuRowMajor, CuRowMajor, CuColMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuRowMajor, CuRowMajor, CuColMajor, CuRowMajor);
				}
			}
			else
			{
				if (orderDest == CuColMajor)
				{
					CALL_TRINARY(CuRowMajor, CuRowMajor, CuRowMajor, CuColMajor);
				}
				else
				{
					CALL_TRINARY(CuRowMajor, CuRowMajor, CuRowMajor, CuRowMajor);
				}
			}
		}
	}

#undef CALL_TRINARY
}


