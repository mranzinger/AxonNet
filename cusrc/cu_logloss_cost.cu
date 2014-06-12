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

#define safe_pred(val) min(max(val, s_epss), 1.0f - s_epss)

CuLoglossCost::CuLoglossCost(int deviceId)
	: _outputIsSoftmax(false)
{
	_handle = CuSetupProvider::GetHandle(deviceId);

	_secondHandle.Device = deviceId;
	cublasCreate_v2(&_secondHandle.CublasHandle);

	cudaStreamCreate(&_secondStream);

	_cacheCompLL = new CuMat(_handle);
	_cacheCompMaxIdxs = new CuMat(_secondHandle);
	_cacheCompBinarized = new CuMat(_secondHandle);

	_cacheCost = new CuMat(_handle);
	_cacheCost->SetSharedModify(true);
}

CuLoglossCost::~CuLoglossCost()
{
	delete _cacheCompLL;
	delete _cacheCompMaxIdxs;
	delete _cacheCompBinarized;
	delete _cacheCost;

	cudaStreamDestroy(_secondStream);

	cublasDestroy_v2(_secondHandle.CublasHandle);
}

struct CuLLVecComputeFn
{
	__device__ Real operator()(Real pred, Real label) const
	{
		Real sPred = safe_pred(pred);

		return label * log(sPred) + (1.0f - label) * log(1 - sPred);
	}
};

struct CuLLVecGradFn
{
	const Real _scale;

	CuLLVecGradFn(const CuMat &labMat)
		: _scale(1.0f / labMat.Cols()) { }

	__device__ Real operator()(Real pred, Real label) const
	{
		Real sPred = safe_pred(pred);

		return (((1.0f - label) / (1.0f - sPred)) - (label / sPred)) * _scale;
	}
};

struct CuLLIdxComputeFn
{
	const Real *_labMat;

	CuLLIdxComputeFn(const CuMat &labMat)
		: _labMat(labMat.Buff()) { }

	__device__ Real operator()(Real pred, uint32_t row, uint32_t col) const
	{
		Real sPred = safe_pred(pred);

		Real labIdx = _labMat[col];

		if (col == labIdx)
			return log(sPred);
		else
			return log(1.0f - sPred);
	}
};

struct CuLLIdxGradFn
{
	const Real _scale;
	const Real *_labMat;

	CuLLIdxGradFn(const CuMat &labMat)
		: _labMat(labMat.Buff()), _scale(1.0f / labMat.Cols()) { }

	__device__ Real operator()(Real pred, uint32_t row, uint32_t col) const
	{
		Real sPred = safe_pred(pred);

		Real labIdx = _labMat[col];

		if (col == labIdx)
		{
			return (-1.0f / sPred) * _scale;
		}
		else
		{
			return (1.0f / (1.0f - sPred)) * _scale;
		}
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

struct CuLLIdxSoftmaxGradFn
{
	const Real *_pLabels;
	const Real _scale;

	CuLLIdxSoftmaxGradFn(const CuMat &labels)
		: _pLabels(labels.Buff()),
		  _scale(1.0f / labels.Cols()) { }

	__device__ Real operator()(Real pred, uint32_t row, uint32_t col) const
	{
		const Real label = _pLabels[col];

		const Real val = (row == label) ? pred - 1.0f : pred;

		const Real scaled = _scale * val;

		return scaled;
	}
};

CostMap CuLoglossCost::Compute(const Params& pred, const Params& labels)
{
	const CuMat &mPred = pred.GetCudaMatrix(_handle);
	const CuMat &mLabels = labels.GetCudaMatrix(_handle);

	Real logLoss, numCorr;

	cudaStreamSynchronize(0);

	// Get the index for the maximum value in each column
	mPred.Colwise().MaxIdx(*_cacheCompMaxIdxs);

	// Index mode. Each label is stored by index
	if (labels.Rows == 1)
	{
		mPred.UnaryExpr<false>(*_cacheCompLL, CuLLIdxComputeFn(mLabels));
		_cacheCompMaxIdxs->BinaryExpr<false>(mLabels, *_cacheCompBinarized, CuLLIdxMaxEqFn());
	}
	// Vector mode
	else
	{
		mPred.BinaryExpr<false>(mLabels, *_cacheCompLL, CuLLVecComputeFn());
		mPred.BinaryExpr<false>(mLabels, *_cacheCompBinarized, CuLLVecMaxEqFn(*_cacheCompMaxIdxs));
	}

	logLoss = _cacheCompLL->Sum();

	numCorr = _cacheCompBinarized->Sum();

	CostMap ret;
	ret[CostMap::PRIMARY_NAME] = logLoss;
	ret["correct"] = numCorr;
	return ret;
}



Params CuLoglossCost::ComputeGrad(const Params& pred, const Params& labels)
{
	const CuMat &mPred = pred.GetCudaMatrix(_handle);
	const CuMat &mLabels = labels.GetCudaMatrix(_handle);

	_cacheCost->ResizeLike(mPred);
	CuMat *cost = new CuMat(*_cacheCost);

	if (_outputIsSoftmax)
	{
		if (labels.Rows == 1)
		{
			mPred.UnaryExpr<false>(*cost, CuLLIdxSoftmaxGradFn(mLabels));
		}
		else
		{
			mPred.BinaryExpr<false>(mLabels, *cost, CuScaledDiff(1.0f / pred.Cols));
			//AddScaled(mPred, 1.0f / pred.Cols, mLabels, -1.0f / pred.Cols, *cost);
		}
	}
	else
	{
		if (labels.Rows == 1)
		{
			mPred.UnaryExpr<false>(*cost, CuLLIdxGradFn(mLabels));
		}
		else
		{
			mPred.BinaryExpr<false>(mLabels, *cost, CuLLVecGradFn(mLabels));
		}
	}

	return Params(pred, cost);
}

void CuLoglossCost::SetOpIsSoftmax(bool value)
{
	_outputIsSoftmax = value;
}


