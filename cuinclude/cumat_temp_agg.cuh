/*
 * File description: cumat_temp_agg.cuh
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "cumat.cuh"

#include <stdint.h>

#include "cudev_helper.cuh"
#include "cumath_functions.cuh"

namespace {

template<uint32_t RowInc, uint32_t ColInc>
class Incrementer
{
public:
	static const bool IsHorizontal = ColInc >= 1;
	static const bool IsVertical = RowInc >= 1;

    __device__ __host__ void operator()(uint32_t &row, uint32_t &col) const
    {
        row += RowInc;
        col += ColInc;
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

template<CuStorageOrder storageOrder, typename Inc, typename Aggregator, typename ElemFn>
__global__ void CuDeviceAggSimple(const CuMatInfo inputMat, CuMatInfo outputMat, Inc inc, Aggregator agg, ElemFn fn)
{
	const uint32_t startRow = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t startCol = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t rows = inputMat._rows;
	const uint32_t cols = inputMat._cols;

	uint32_t row = startRow;
	uint32_t col = startCol;

	// Check the boundary
	if (row >= inputMat._rows || col >= inputMat._cols)
		return;

	const Real *buff = inputMat._dMat;

	// Initialize the accumulator to the initial value
	Real accum = fn(buff[ElementIdx<storageOrder>(row, col, rows, cols)]);
	inc(row, col);

	for (; (Inc::IsHorizontal && col < cols) || (Inc::IsVertical && row < rows);
	     inc(row, col))
	{
		Real comp = fn(buff[ElementIdx<storageOrder>(row, col, rows, cols)]);

		accum = agg(accum, comp);
	}

	// Store the value
	uint32_t opIdx = ElementIdx<storageOrder>(startRow, startCol, outputMat._rows, outputMat._cols);

	outputMat._dMat[opIdx] = accum;
}

template<typename Inc, typename Aggregator, typename ElemFn>
void CuMatAggregate(const CuMat &mat, CuMat &dest,
				    Inc inc, Aggregator agg, ElemFn fn,
				    cublasHandle_t cublasHandle)
{
	if (!cublasHandle)
		cublasHandle = mat.Handle().CublasHandle;

	cudaStream_t stream;
	cublasGetStream_v2(cublasHandle, &stream);

	// This will probably be the simplest (and slowest) way to implement this
	uint32_t xDim = Inc::XDim(mat.Cols()),
			 yDim = Inc::YDim(mat.Rows());

	dim3 blockSize = round_up<32>(xDim, yDim);
	dim3 threadSize(min(32u, xDim), min(32u, yDim));

	dest.Resize(yDim, xDim);

	if (mat.Order() == CuColMajor)
	{
		CuDeviceAggSimple<CuColMajor><<<blockSize, threadSize, 0, stream>>>(mat, dest, inc, agg, fn);
	}
	else
	{
		CuDeviceAggSimple<CuRowMajor><<<blockSize, threadSize, 0, stream>>>(mat, dest, inc, agg, fn);
	}
}

template<typename Inc, typename Aggregator, typename ElemFn>
CuMat CuMatAggregate(const CuMat &mat,
                     Inc inc, Aggregator agg, ElemFn fn)
{
	CuMat ret(mat.Handle());

	CuMatAggregate(mat, ret, inc, agg, fn, mat.Handle().CublasHandle);

	return ret;
}



}

template<typename ElemFn>
CuMat CuRowwiseOperator::Sum(ElemFn fn) const
{
    return Agg(CuPlus(), fn);
}

template<typename ElemFn>
void CuRowwiseOperator::Sum(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle) const
{
	Agg(dest, CuPlus(), fn, cublasHandle);
}

template<typename ElemFn>
CuMat CuColwiseOperator::Sum(ElemFn fn) const
{
    return Agg(CuPlus(), fn);
}

template<typename ElemFn>
void CuColwiseOperator::Sum(CuMat &dest, ElemFn fn, cublasHandle_t cublasHandle) const
{
	Agg(dest, CuPlus(), fn, cublasHandle);
}

template<typename ElemFn>
CuMat CuRowwiseOperator::Max(ElemFn fn) const
{
	return Agg(CuMax(), fn);
}

template<typename ElemFn>
CuMat CuColwiseOperator::Max(ElemFn fn) const
{
	return Agg(CuMax(), fn);
}

template<typename ElemFn>
CuMat CuRowwiseOperator::Min(ElemFn fn) const
{
	return Agg(CuMin(), fn);
}

template<typename ElemFn>
CuMat CuColwiseOperator::Min(ElemFn fn) const
{
	return Agg(CuMin(), fn);
}

template<typename Aggregator, typename ElemFn>
CuMat CuRowwiseOperator::Agg(Aggregator agg, ElemFn fn) const
{
    return CuMatAggregate(Mat,
                          Incrementer<0, 1>(),
                          agg, fn);
}

template<typename Aggregator, typename ElemFn>
void CuRowwiseOperator::Agg(CuMat &dest, Aggregator agg, ElemFn fn,
							cublasHandle_t cublasHandle) const
{
	CuMatAggregate(Mat, dest, Incrementer<0, 1>(),
				   agg, fn, cublasHandle);
}

template<typename Aggregator, typename ElemFn>
CuMat CuColwiseOperator::Agg(Aggregator agg, ElemFn fn) const
{
    return CuMatAggregate(Mat,
                          Incrementer<1, 0>(),
                          agg, fn);
}

template<typename Aggregator, typename ElemFn>
void CuColwiseOperator::Agg(CuMat &dest, Aggregator agg, ElemFn fn,
							cublasHandle_t cublasHandle) const
{
	CuMatAggregate(Mat, dest, Incrementer<1, 0>(),
				   agg, fn, cublasHandle);
}


