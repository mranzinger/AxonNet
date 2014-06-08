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
	// Get the differences and square them
	CuMat diff = pred.GetCudaMatrix(_handle) - labels.GetCudaMatrix(_handle);
	diff.UnaryExpr(CuSquare());

	Real cost = diff.Sum();

	CostMap ret;
	ret[CostMap::PRIMARY_NAME] = cost;

	return ret;
}

Params CuSumSqCost::ComputeGrad(const Params& pred, const Params& labels)
{
	CuMat *diff = new CuMat(_handle);

	*diff = pred.GetCudaMatrix(_handle) - labels.GetCudaMatrix(_handle);

	(*diff) /= pred.Cols;

	return Params(pred, diff);
}
