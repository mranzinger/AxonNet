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

template<CuStorageOrder storageOrder, typename Inc, typename Aggregator, typename ElemFn>
__global__ void CuDeviceAgg(const CuMatInfo inputMat, CuMatInfo outputMat, Inc inc, Aggregator agg, ElemFn fn)
{
    // Use shared memory to store the intermediate results for each thread
    extern __shared__ Real s_results[];

    const uint32_t startRow = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t startCol = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = inputMat._rows;
    const uint32_t cols = inputMat._cols;

    uint32_t row = startRow;
    uint32_t col = startCol;

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

    s_results[tid] = agg.NullValue();

    if (row >= inputMat._rows || col >= inputMat._cols)
        return;

    uint32_t blockSize = blockDim.x * blockDim.y;

    const Real *ipBuff = inputMat._dMat;

    // First stage: Accumulate locally into the shared memory
    Real accum = fn(ipBuff[ElementIdx<storageOrder>(row, col, rows, cols)]);
    inc(row, col, blockSize);

    for (; (Inc::IsHorizontal && col < cols) || (Inc::IsVertical && row < rows);
         inc(row, col, blockSize))
    {
        Real comp = fn(ipBuff[ElementIdx<storageOrder>(row, col, rows, cols)]);

        accum = agg(accum, comp);
    }

    // Store the intermediate result in the allocated shared memory spot.
    s_results[tid] = accum;

    // Wait for all of the threads to finish their initial aggregation
    __syncthreads();

    // Now accumulate the shared values into a single value
    for (uint32_t s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_results[tid] = agg(s_results[tid], s_results[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // Store the value
        uint32_t opIdx = ElementIdx<storageOrder>(startRow, startCol, outputMat._rows, outputMat._cols);

        outputMat._dMat[opIdx] = s_results[0];
    }
}

template<typename Inc, typename Aggregator, typename ElemFn>
void CuMatAggregate(const CuMat &mat, CuMat &dest,
				    Inc inc, Aggregator agg, ElemFn fn,
				    cublasHandle_t cublasHandle)
{
    static const uint32_t s_assumedCompute = 1024;

    if (mat.Empty())
        return;

	if (!cublasHandle)
		cublasHandle = mat.Handle().CublasHandle;

	cudaStream_t stream;
	cublasGetStream_v2(cublasHandle, &stream);

	// This will probably be the simplest (and slowest) way to implement this
	uint32_t xDim = Inc::XDim(mat.Cols()),
			 yDim = Inc::YDim(mat.Rows());

	// Compute the number of elements that each thread block will need to process
	uint32_t elsPerThreadBlock = mat.Size() / (xDim * yDim);
	// Estimate the number of threads that are likely available to do this processing
	uint32_t threadsPerBlock = min(max(s_assumedCompute / (xDim * yDim), 1u), 512u);

	// Now compute the number of elements that each thread should process
	//uint32_t elsPerThread = round_up(elsPerThreadBlock, threadsPerBlock);

	dim3 blockSize, gridSize;
	if (Inc::IsHorizontal)
	{
	    blockSize = dim3(threadsPerBlock, 1);
	    gridSize = dim3(1, yDim);
	}
	else
	{
	    blockSize = dim3(1, threadsPerBlock);
	    gridSize = dim3(xDim, 1);
	}

	uint32_t smemSize = threadsPerBlock * sizeof(Real);

	dest.Resize(yDim, xDim);

	if (mat.Order() == CuColMajor)
	{
	    CuDeviceAgg<CuColMajor>
#ifdef _CUDA_COMPILE_
	        <<<gridSize, blockSize, smemSize, stream>>>
#endif
	        (mat, dest, inc, agg, fn);
	}
	else
	{
        CuDeviceAgg<CuRowMajor>
#ifdef _CUDA_COMPILE_
            <<<gridSize, blockSize, smemSize, stream>>>
#endif
            (mat, dest, inc, agg, fn);
	}

	/*dim3 blockSize = round_up<32>(xDim, yDim);
	dim3 threadSize(min(32u, xDim), min(32u, yDim));

	if (mat.Order() == CuColMajor)
	{
		CuDeviceAggSimple<CuColMajor><<<blockSize, threadSize, 0, stream>>>(mat, dest, inc, agg, fn);
	}
	else
	{
		CuDeviceAggSimple<CuRowMajor><<<blockSize, threadSize, 0, stream>>>(mat, dest, inc, agg, fn);
	}*/
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


