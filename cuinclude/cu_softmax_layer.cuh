/*
 * cu_softmax_layer.cuh
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */


#pragma once

#include "params.h"

class CuSoftmaxLayer
{
scope_public:
	CuSoftmaxLayer(int deviceId);
	~CuSoftmaxLayer();

	Params Compute(const Params &input);
	Params Backprop(const Params &lastInput, const Params &lastOutput,
					const Params &outputErrors);

	void SetCostIsLogreg(bool value);

scope_private:
	bool _costIsLogreg;
	CuContext _handle;

	CuMat *_cacheCompute;
	CuMat *_cacheBackprop;
	CuMat *_cacheIpMax;
	CuMat *_cacheExpSum;
	//CuMat *_cacheJacobian;
};
