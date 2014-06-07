/*
 * cu_softmax_layer.cu
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */

#include "cu_softmax_layer.cuh"

#include "cusetup_provider.cuh"
#include "cumat.cuh"

struct CuSoftmaxExpr
{
	const Real *_maxBuff;

	CuSoftmaxExpr(const CuMat &mat)
		: _maxBuff(mat.Buff()) { }

	__device__ Real operator()(Real value, uint32_t row, uint32_t col) const
	{
		return exp(value - _maxBuff[col]);
	}
};

struct CuSoftmaxDiv
{
    const Real *_invSumBuff;

    CuSoftmaxDiv(const CuMat &mat)
        : _invSumBuff(mat.Buff()) { }

    __device__ Real operator()(Real value, uint32_t row, uint32_t col) const
    {
        return value * _invSumBuff[col];
    }
};

void CalcSoftmaxDiff(CuMat &mDiff, const CuMat &lastOutput);

CuSoftmaxLayer::CuSoftmaxLayer(int deviceId)
	: _costIsLogreg(false)
{
	_handle = CuSetupProvider::GetHandle(deviceId);
}

Params CuSoftmaxLayer::Compute(const Params& input) const
{
	CuMat *m = new CuMat(_handle, input.Rows, input.Cols);

	Params ret(input, m);

	const CuMat &mInput = input.GetCudaMatrix(_handle);

	// Get the maximum value in each column
	CuMat ipMax = mInput.Colwise().Max();

	CuMat &mSoftmax = *m;
	mInput.UnaryExpr<false>(mSoftmax, CuSoftmaxExpr(ipMax));

	// Sum the columns, and also take their inverse to make the
	// subsequent operations faster
	CuMat mExpMatSum = mSoftmax.Colwise().Sum();
	mExpMatSum.UnaryExpr(CuInverse());

	// Now divide all of the elements by the columnar sums
	mSoftmax.UnaryExpr(CuSoftmaxDiv(mExpMatSum));

	// Need to synchronize since streams are allocated
	cudaDeviceSynchronize();

	return ret;
}

Params CuSoftmaxLayer::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors) const
{
	if (_costIsLogreg)
		// The cost function already computed the input error for this guy
		return outputErrors;

	Params ret = Params::CreateLike(lastInput, _handle);

	CuMat &inputErrors = ret.GetCudaMatrix(_handle);

	// Create a big jacobian matrix of first derivatives
	CuMat mDiff(_handle, lastOutput.Rows * lastOutput.Rows, lastOutput.Cols);
	CalcSoftmaxDiff(mDiff, lastOutput.GetCudaMatrix(_handle));

	MultiplyTrans3D(mDiff, lastOutput.Rows, lastOutput.Rows,
				    outputErrors.GetCudaMatrix(_handle), inputErrors);

	cudaDeviceSynchronize();

	return ret;
}

void CuSoftmaxLayer::SetCostIsLogreg(bool value)
{
	_costIsLogreg = value;
}


__global__ void CuCalcSoftmaxDiff(CuMatInfo mDiffInfo, CuMatInfo mLastOpInfo)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	// Rows both times here is intentional
	if (x >= mLastOpInfo._rows || y >= mLastOpInfo._rows)
		return;

	uint32_t opRow = y * mLastOpInfo._rows + x;

	uint32_t opIdx = z * mDiffInfo._rows + opRow;

	uint32_t ipOffset = z * mLastOpInfo._rows;

	Real dX = mLastOpInfo._dMat[ipOffset + x];
	Real dY = mLastOpInfo._dMat[ipOffset + y];

	Real dXdY = dY * ((x == y) - dX);

	mDiffInfo._dMat[opIdx] = dXdY;
}

void CalcSoftmaxDiff(CuMat& mDiff, const CuMat& lastOutput)
{
	dim3 threads(min(32u, lastOutput.Rows()),
				 min(32u, lastOutput.Rows()),
				 1);
	dim3 blocks = round_up(lastOutput.Rows(), lastOutput.Rows(), lastOutput.Cols(),
						   threads);

	// TODO: The matrix is symmetric, so a single block could absolutely scatter
	// its output into the corresponding x,y pairs
	CuCalcSoftmaxDiff<<<blocks, threads>>>(mDiff, lastOutput);
}



























