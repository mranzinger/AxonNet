/*
 * cu_softmax_layer.cu
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */

#include "cu_softmax_layer.cuh"

#include "cusetup_provider.cuh"

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

	return ret;
}

Params CuSoftmaxLayer::Backprop(const Params& lastInput,
		const Params& lastOutput, const Params& outputErrors) const
{
	CuMat *m = new CuMat(_handle, lastInput.Rows, lastInput.Cols);

	Params ret(lastInput, m);

	if (_costIsLogreg)
	{
		// The cost function already
	}
	else
	{

	}

	return ret;
}

void CuSoftmaxLayer::SetCostIsLogreg(bool value)
{
	_costIsLogreg = value;
}
