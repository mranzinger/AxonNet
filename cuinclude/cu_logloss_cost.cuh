/*
 * cu_logloss_cost.cuh
 *
 *  Created on: Jun 8, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"
#include "cost_map.h"

class CuLoglossCost
{
scope_public:
	CuLoglossCost(int deviceId);
	~CuLoglossCost();

	CostMap Compute(const Params &pred, const Params &labels);
	Params ComputeGrad(const Params &pred, const Params &labels);

	void SetOpIsSoftmax(bool value);

scope_private:
	CuContext _handle;
	CuContext _secondHandle;
	bool _outputIsSoftmax;

	cudaStream_t _secondStream;
	CuMat *_cacheCompLL;
	CuMat *_cacheCompMaxIdxs;
	CuMat *_cacheCompBinarized;

	CuMat *_cacheCost;
};


