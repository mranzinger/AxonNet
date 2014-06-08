/*
 * cu_logloss_cost.cu
 *
 *  Created on: Jun 8, 2014
 *      Author: mike
 */

#include "cu_logloss_cost.cuh"

#include "cusetup_provider.cuh"
#include "cumat.cuh"

__device__ __constant__ Real s_epss = 0.000000001;

CuLoglossCost::CuLoglossCost(int deviceId)
	: _outputIsSoftmax(false)
{
	_handle = CuSetupProvider::GetHandle(deviceId);
}

struct CuLLVecComputeFn
{
	__device__ Real operator()(Real pred, Real label) const
	{
		Real sPred = min(max(pred, s_epss), 1.0f - s_epss);

		return label * log(sPred) + (1.0f - label) * log(1 - sPred);
	}
};

struct CuLLIdxComputeFn
{
	const Real *_labMat;

	CuLLIdxComputeFn(const CuMat &labMat)
		: _labMat(labMat.Buff()) { }

	__device__ Real operator()(Real pred, uint32_t row, uint32_t col) const
	{
		Real sPred = min(max(pred, s_epss), 1.0f - s_epss);

		Real labIdx = _labMat[col];

		if (col == labIdx)
			return log(sPred);
		else
			return log(1.0f - sPred);
	}
};

struct CuLLVecMaxEqFn
{
	const Real *_maxIdx;

	CuLLVecMaxEqFn(const CuMat &maxIdxMat)
		: _maxIdx(maxIdxMat.Buff()) { }

	__device__ Real operator()(Real pred, Real label, uint32_t row, uint32_t col) const
	{
		if (label == 0.0f)
			return 0.0f;

		Real mIdx = _maxIdx[col];

		if (mIdx == row)
			return 1.0f;
		else
			return 0.0f;
	}
};

struct CuLLIdxMaxEqFn
{
	__device__ Real operator()(Real maxIdx, Real labelIdx) const
	{
		return maxIdx == labelIdx;
	}
};

CostMap CuLoglossCost::Compute(const Params& pred, const Params& labels)
{
	const CuMat &mPred = pred.GetCudaMatrix(_handle);
	const CuMat &mLabels = labels.GetCudaMatrix(_handle);

	Real logLoss, numCorr;
	{
		CuMat ll(_handle);

		// Index mode. Each label is stored by index
		if (labels.Rows == 1)
		{
			mPred.UnaryExpr<false>(ll, CuLLIdxComputeFn(mLabels));
		}
		// Vector mode
		else
		{
			mPred.BinaryExpr<false>(mLabels, ll, CuLLVecComputeFn());
		}

		logLoss = ll.Sum();
	}
	{
		// Get the index for the maximum value in each column
		CuMat maxIdxs = mPred.Colwise().MaxIdx();

		CuMat bin(_handle);

		if (labels.Rows == 1)
		{
			maxIdxs.BinaryExpr<false>(mLabels, bin, CuLLIdxMaxEqFn());
		}
		else
		{
			mPred.BinaryExpr<false>(mLabels, bin, CuLLVecMaxEqFn(maxIdxs));
		}

		numCorr = bin.Sum();
	}

	CostMap ret;
	ret[CostMap::PRIMARY_NAME] = logLoss;
	ret["correct"] = numCorr;
	return ret;
}

Params CuLoglossCost::ComputeGrad(const Params& pred, const Params& labels)
{

}

void CuLoglossCost::SetOpIsSoftmax(bool value)
{
	_outputIsSoftmax = value;
}
