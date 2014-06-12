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
    extern __shared__ Real s_valAgg[];

    const uint32_t startRow = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t startCol = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = inputMat._rows;
    const uint32_t cols = inputMat._cols;

    uint32_t row = startRow;
    uint32_t col = startCol;

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

    s_valAgg[tid] = agg.NullValue();

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
    s_valAgg[tid] = accum;

    // Wait for all of the threads to finish their initial aggregation
    __syncthreads();

    // Now accumulate the shared values into a single value
    for (uint32_t s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_valAgg[tid] = agg(s_valAgg[tid], s_valAgg[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // Store the value
        uint32_t opIdx = ElementIdx<storageOrder>(startRow, startCol, outputMat._rows, outputMat._cols);

        outputMat._dMat[opIdx] = s_valAgg[0];
    }
}



template<CuStorageOrder storageOrder, typename Inc, typename Aggregator, typename ElemFn>
__global__ void CuDeviceAggIdx(const CuMatInfo inputMat, CuMatInfo outputMat, Inc inc, Aggregator agg, ElemFn fn)
{
    // Use shared memory to store the intermediate results for each thread
    extern __shared__ ValIdx s_idxAgg[];

    const uint32_t startRow = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t startCol = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = inputMat._rows;
    const uint32_t cols = inputMat._cols;

    uint32_t row = startRow;
    uint32_t col = startCol;

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

    s_idxAgg[tid] = ValIdx(agg.NullValue(), tid);

    if (row >= inputMat._rows || col >= inputMat._cols)
        return;

    uint32_t blockSize = blockDim.x * blockDim.y;

    const Real *ipBuff = inputMat._dMat;

    uint32_t startIdx;
    if (Inc::IsHorizontal)
    	startIdx = col;
    else
    	startIdx = row;

    // First stage: Accumulate locally into the shared memory
    ValIdx accum(fn(ipBuff[ElementIdx<storageOrder>(row, col, rows, cols)]), startIdx);
    inc(row, col, blockSize);

    for (uint32_t idx = startIdx + blockSize;
    	  (Inc::IsHorizontal && col < cols) || (Inc::IsVertical && row < rows);
         inc(row, col, blockSize), idx += blockSize)
    {
        ValIdx comp(fn(ipBuff[ElementIdx<storageOrder>(row, col, rows, cols)]), idx);

        accum = agg(accum, comp);
    }

    // Store the intermediate result in the allocated shared memory spot.
    s_idxAgg[tid] = accum;

    // Wait for all of the threads to finish their initial aggregation
    __syncthreads();

    // Now accumulate the shared values into a single value
    for (uint32_t s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_idxAgg[tid] = agg(s_idxAgg[tid], s_idxAgg[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // Store the value
        uint32_t opIdx = ElementIdx<storageOrder>(startRow, startCol, outputMat._rows, outputMat._cols);

        outputMat._dMat[opIdx] = s_idxAgg[0].Idx;
    }
}

template<typename Impl>
struct CuMatAggregator_t
{
    template<typename Inc, typename Aggregator, typename ElemFn>
    static void Invoke(const CuMat &mat, CuMat &dest,
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

        dest.Resize(yDim, xDim);

        if (mat.Order() == CuColMajor)
        {
            Impl::template Call<CuColMajor>(mat, dest,
                                     inc, agg, fn,
                                     cublasHandle,
                                     gridSize, blockSize,
                                     stream);
        }
        else
        {
            Impl::template Call<CuRowMajor>(mat, dest,
                                     inc, agg, fn,
                                     cublasHandle,
                                     gridSize, blockSize,
                                     stream);
        }
    }
};

template<bool Vals>
struct CuMatAggregator
    : CuMatAggregator_t<CuMatAggregator<Vals> >
{
    template<CuStorageOrder order, typename Inc, typename Aggregator, typename ElemFn>
    static void Call(const CuMat &mat, CuMat &dest,
                       Inc inc, Aggregator agg, ElemFn fn,
                       cublasHandle_t cublasHandle,
                       const dim3 &gridSize, const dim3 &blockSize,
                       cudaStream_t stream)
    {
        // Aggregate values
        uint32_t smemSize = (blockSize.x * blockSize.y) * sizeof(Real);

        CuDeviceAgg<order>
            <<<gridSize, blockSize, smemSize, stream>>>
            (mat, dest, inc, agg, fn);
    }
};

template<>
struct CuMatAggregator<false>
    : CuMatAggregator_t<CuMatAggregator<false> >
{
    template<CuStorageOrder order, typename Inc, typename Aggregator, typename ElemFn>
    static void Call(const CuMat &mat, CuMat &dest,
                       Inc inc, Aggregator agg, ElemFn fn,
                       cublasHandle_t cublasHandle,
                       const dim3 &gridSize, const dim3 &blockSize,
                       cudaStream_t stream)
    {
        // Aggregate indexes
        uint32_t smemSize = (blockSize.x * blockSize.y) * sizeof(ValIdx);

        CuDeviceAggIdx<order>
            <<<gridSize, blockSize, smemSize, stream>>>
            (mat, dest, inc, agg, fn);
    }
};

template<bool Vals, typename Inc, typename Aggregator, typename ElemFn>
void CuMatAggregate(const CuMat &mat, CuMat &dest,
				    Inc inc, Aggregator agg, ElemFn fn,
				    cublasHandle_t cublasHandle)
{
    CuMatAggregator<Vals>::Invoke(mat, dest,
                                  inc, agg, fn,
                                  cublasHandle);
}

template<bool Vals, typename Inc, typename Aggregator, typename ElemFn>
CuMat CuMatAggregate(const CuMat &mat,
                     Inc inc, Aggregator agg, ElemFn fn)
{
	CuMat ret(mat.Handle());

	CuMatAggregate<Vals>(mat, ret, inc, agg, fn, mat.Handle().CublasHandle);

	return ret;
}

}

template<typename Inc>
CuMat CuMatAgg_t<Inc>::Sum() const
{
    return Sum(CuIdentity());
}

template<typename Inc>
template<typename ElemFn>
CuMat CuMatAgg_t<Inc>::Sum(ElemFn fn) const
{
    return Agg<true>(CuPlus(), fn);
}

template<typename Inc>
void CuMatAgg_t<Inc>::Sum(CuMat& dest, cublasHandle_t cublasHandle) const
{
    Sum(dest, CuIdentity(), cublasHandle);
}

template<typename Inc>
template<typename ElemFn>
void CuMatAgg_t<Inc>::Sum(CuMat& dest, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    Agg<true>(dest, CuPlus(), fn, cublasHandle);
}

template<typename Inc>
CuMat CuMatAgg_t<Inc>::Max() const
{
    return Max(CuIdentity());
}

template<typename Inc>
template<typename ElemFn>
CuMat CuMatAgg_t<Inc>::Max(ElemFn fn) const
{
    return Agg<true>(CuMax(), fn);
}

template<typename Inc>
void CuMatAgg_t<Inc>::Max(CuMat& dest, cublasHandle_t cublasHandle) const
{
    Max(dest, CuIdentity(), cublasHandle);
}

template<typename Inc>
template<typename ElemFn>
void CuMatAgg_t<Inc>::Max(CuMat& dest, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    Agg<true>(dest, CuMax(), fn, cublasHandle);
}

template<typename Inc>
CuMat CuMatAgg_t<Inc>::MaxIdx() const
{
    return MaxIdx(CuIdentity());
}

template<typename Inc>
template<typename ElemFn>
CuMat CuMatAgg_t<Inc>::MaxIdx(ElemFn fn) const
{
    return Agg<false>(CuMax(), fn);
}

template<typename Inc>
void CuMatAgg_t<Inc>::MaxIdx(CuMat& dest,
        cublasHandle_t cublasHandle) const
{
    MaxIdx(dest, CuIdentity(), cublasHandle);
}

template<typename Inc>
template<typename ElemFn>
void CuMatAgg_t<Inc>::MaxIdx(CuMat& dest, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    Agg<false>(dest, CuMax(), fn, cublasHandle);
}

template<typename Inc>
CuMat CuMatAgg_t<Inc>::Min() const
{
    return Min(CuIdentity());
}

template<typename Inc>
template<typename ElemFn>
CuMat CuMatAgg_t<Inc>::Min(ElemFn fn) const
{
    return Agg<true>(CuMin(), fn);
}

template<typename Inc>
void CuMatAgg_t<Inc>::Min(CuMat& dest, cublasHandle_t cublasHandle) const
{
    Min(dest, CuIdentity(), cublasHandle);
}

template<typename Inc>
template<typename ElemFn>
void CuMatAgg_t<Inc>::Min(CuMat& dest, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    Agg(dest, CuMin(), fn, cublasHandle);
}

template<typename Inc>
CuMat CuMatAgg_t<Inc>::MinIdx() const
{
    return MinIdx(CuIdentity());
}

template<typename Inc>
template<typename ElemFn>
CuMat CuMatAgg_t<Inc>::MinIdx(ElemFn fn) const
{
    return Agg<false>(CuMin(), fn);
}

template<typename Inc>
void CuMatAgg_t<Inc>::MinIdx(CuMat& dest,
        cublasHandle_t cublasHandle) const
{
    MinIdx(dest, CuIdentity(), cublasHandle);
}

template<typename Inc>
template<typename ElemFn>
void CuMatAgg_t<Inc>::MinIdx(CuMat& dest, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    Agg<false>(dest, CuMin(), fn, cublasHandle);
}

template<typename Inc>
template<bool Vals, typename Aggregator, typename ElemFn>
CuMat CuMatAgg_t<Inc>::Agg(Aggregator agg, ElemFn fn) const
{
    return CuMatAggregate<Vals>(Mat, Inc(), agg, fn);
}

template<typename Inc>
template<bool Vals, typename Aggregator, typename ElemFn>
void CuMatAgg_t<Inc>::Agg(CuMat& dest, Aggregator agg, ElemFn fn,
        cublasHandle_t cublasHandle) const
{
    CuMatAggregate<Vals>(Mat, dest, Inc(), agg, fn, cublasHandle);
}


