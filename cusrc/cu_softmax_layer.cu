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

CuSoftmaxLayer::CuSoftmaxLayer(int deviceId)
	: _costIsLogreg(false)
{
	_handle = CuSetupProvider::GetHandle(deviceId);

	_cacheCompute = new CuMat(_handle);
	_cacheCompute->SetSharedModify(true);

	_cacheBackprop = new CuMat(_handle);
	_cacheBackprop->SetSharedModify(true);

	_cacheIpMax = new CuMat(_handle);
	_cacheExpSum = new CuMat(_handle);
}

CuSoftmaxLayer::~CuSoftmaxLayer()
{
    delete _cacheCompute;
    delete _cacheBackprop;
    delete _cacheIpMax;
    delete _cacheExpSum;
}

Params CuSoftmaxLayer::Compute(const Params& input)
{
	//CuMat *m = new CuMat(_handle, input.Rows, input.Cols);
    _cacheCompute->Resize(input.Rows, input.Cols);
    CuMat *m = new CuMat(*_cacheCompute);

	Params ret(input, m);

	const CuMat &mInput = input.GetCudaMatrix(_handle);

	// Get the maximum value in each column
	mInput.Colwise().Max(*_cacheIpMax);

	CuMat &mSoftmax = *m;
	mInput.UnaryExpr<false>(mSoftmax, CuSoftmaxExpr(*_cacheIpMax));

	// Sum the columns, and also take their inverse to make the
	// subsequent operations faster
	mSoftmax.Colwise().Sum(*_cacheExpSum);
	_cacheExpSum->UnaryExpr(CuInverse());

	// Now divide all of the elements by the columnar sums
	mSoftmax.UnaryExpr(CuSoftmaxDiv(*_cacheExpSum));

	return ret;
}

__global__ void CuSoftmaxLayer_CalcInputErrors(
                    const Real *gLastOutput,
                    const Real *gOutputErrors,
                    Real *gInputErrors,
                    const uint32_t numRows)
{
    // TODO: Use shared memory
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= numRows)
        return;

    const uint32_t layer = blockIdx.z * blockDim.z + threadIdx.z;

    const Real *lLastOutput = gLastOutput + layer * numRows;
    const Real *lOutputErrors = gOutputErrors + layer * numRows;
    Real *lInputErrors = gInputErrors + layer * numRows;

    const uint32_t vecProcX = numRows & ~0x7;

    const Real opY = lLastOutput[y];

    Real sum = 0.0f;
    for (uint32_t x = 0; x < vecProcX; )
    {
        const uint32_t xEnd = x + 8;

        #pragma unroll
        for (; x < xEnd; ++x)
        {
            // This is where shared memory is key...
            const Real opX = lLastOutput[x];

            const Real opErr = lOutputErrors[x];

            assert(opX >= 0);

            const Real kronecker = (x == y) ? 1.0f : 0.0f;

            const Real prod = (opY * (kronecker - opX)) * opErr;

            sum += prod;
        }
    }

    for (uint32_t x = vecProcX; x < numRows; ++x)
    {
        const Real opX = lLastOutput[x];

        assert(opX >= 0);

        const Real kronecker = (x == y) ? 1.0f : 0.0f;

        const Real prod = opY * (kronecker - opX);

        sum += prod;
    }

    lInputErrors[y] = sum;
}

Params CuSoftmaxLayer::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors)
{
	if (_costIsLogreg)
		// The cost function already computed the input error for this guy
		return outputErrors;

	_cacheBackprop->Resize(lastInput.Rows, lastInput.Cols);
	CuMat *m = new CuMat(*_cacheBackprop);

	Params ret(lastInput, m);

	const CuMat &mLastOutput = lastOutput.GetCudaMatrix(_handle);
	const CuMat &mOutputErrors = outputErrors.GetCudaMatrix(_handle);
	CuMat &mInputErrors = ret.GetCudaMatrix(_handle);

	// This computation requires a 3D Jacobian to be computed.
	// To conserve memory (a 90k x 90k matrix is daunting),
	// we will actually directly compute the matrix product while
	// computing the jacobian. This circumvents the matrix being
	// instantiated and hoggin RAM

	dim3 blockSize(1, min(128, lastOutput.Rows), 1);
	dim3 gridSize = round_up(1, mLastOutput.Rows(), mLastOutput.Cols(), blockSize);

	CuSoftmaxLayer_CalcInputErrors<<<gridSize, blockSize>>>
	                              (mLastOutput.Buff(),
	                               mOutputErrors.Buff(),
	                               mInputErrors.Buff(),
	                               mLastOutput.Rows()
	                               );

	return ret;
}

void CuSoftmaxLayer::SetCostIsLogreg(bool value)
{
	_costIsLogreg = value;
}



























