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

	_cacheCompute = new CuMat(_handle);
	_cacheCompute->SetSharedModify(true);

	_cacheBackprop = new CuMat(_handle);
	_cacheBackprop->SetSharedModify(true);

	_cacheIpMax = new CuMat(_handle);
	_cacheExpSum = new CuMat(_handle);
	_cacheJacobian = new CuMat(_handle);
}

CuSoftmaxLayer::~CuSoftmaxLayer()
{
    delete _cacheCompute;
    delete _cacheBackprop;
    delete _cacheIpMax;
    delete _cacheExpSum;
    delete _cacheJacobian;
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

Params CuSoftmaxLayer::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors)
{
	if (_costIsLogreg)
		// The cost function already computed the input error for this guy
		return outputErrors;

	_cacheBackprop->Resize(lastInput.Rows, lastInput.Cols);
	CuMat *m = new CuMat(*_cacheBackprop);

	Params ret(lastInput, m);

	CuMat &inputErrors = ret.GetCudaMatrix(_handle);

	// Create a big jacobian matrix of first derivatives
	_cacheJacobian->Resize(lastOutput.Rows * lastOutput.Rows, lastOutput.Cols);
	CalcSoftmaxDiff(*_cacheJacobian, lastOutput.GetCudaMatrix(_handle));

	MultiplyTrans3D(*_cacheJacobian, lastOutput.Rows, lastOutput.Rows,
				    outputErrors.GetCudaMatrix(_handle), inputErrors);

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



























