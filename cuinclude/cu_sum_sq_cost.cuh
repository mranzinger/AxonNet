/*
 * cu_sumsq_cost.cuh
 *
 *  Created on: Jun 7, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"

class CuSumSqCost
{
scope_public:
	CuSumSqCost(int deviceId);

	CostMap Compute(const Params &pred, const Params &labels);
	Params ComputeGrad(const Params &pred, const Params &labels);

scope_private:
	CuContext _handle;
};


