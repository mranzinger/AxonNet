/*
 * cu_sum_sq_cost.cu
 *
 *  Created on: Jun 7, 2014
 *      Author: mike
 */

#include "cu_sum_sq_cost.cuh"

#include "cusetup_provider.cuh"
#include "cumat.cuh"

CuSumSqCost::CuSumSqCost(int deviceId)
{
	_handle = CuSetupProvider::GetHandle(deviceId);
}

CostMap CuSumSqCost::Compute(const Params& pred, const Params& labels)
{
    const CuMat &mPreds = pred.GetCudaMatrix(_handle);
    const CuMat &mLabels = labels.GetCudaMatrix(_handle);

    // Get the differences and square them
    CuMat diff(_handle);
    mPreds.BinaryExpr<false>(mLabels, diff, CuSquaredDiff());

	Real cost = diff.Sum();

	CostMap ret;
	ret[CostMap::PRIMARY_NAME] = cost;

	return ret;
}

Params CuSumSqCost::ComputeGrad(const Params& pred, const Params& labels)
{
	const CuMat &mPreds = pred.GetCudaMatrix(_handle);
	const CuMat &mLabels = labels.GetCudaMatrix(_handle);

	CuMat *diff = new CuMat(_handle);

	mPreds.BinaryExpr<false>(mLabels, *diff, CuScaledDiff(1.0f / pred.Cols));

	return Params(pred, diff);
}
