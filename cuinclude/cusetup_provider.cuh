/*
 * cusetup_provider.cuh
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */


#pragma once

#include <map>

#include <cublas_v2.h>
#include <cuda.h>

struct CuContext
{
	int Device;
	cublasHandle_t CublasHandle;

	CuContext() : Device(0), CublasHandle(0) { }
};

class CuSetupProvider
{
private:
	CuSetupProvider();
	~CuSetupProvider();

public:
	static CuContext GetHandle(int deviceId, int threadId = 0);

private:
	static CuSetupProvider &Instance();

	CuContext p_GetHandle(int deviceId, int threadId);

	typedef std::map<std::pair<int,int>, CuContext> HandleMap;
	HandleMap _handleMap;
};


