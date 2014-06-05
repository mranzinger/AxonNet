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

#include "math_defines.h"

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


