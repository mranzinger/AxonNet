/*
 * cusetup_provider.cu
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */

#include "cusetup_provider.cuh"

CuSetupProvider::CuSetupProvider()
{
}

CuSetupProvider::~CuSetupProvider()
{
	for (HandleMap::iterator iter = _handleMap.begin(),
			                 end = _handleMap.end();
			iter != end;
			++iter)
	{
		cublasDestroy_v2(iter->second.CublasHandle);
	}
}

CuContext CuSetupProvider::GetHandle(int deviceId, int threadId)
{
	return Instance().p_GetHandle(deviceId, threadId);
}

CuSetupProvider& CuSetupProvider::Instance()
{
	static CuSetupProvider s_provider;

	return s_provider;
}

CuContext CuSetupProvider::p_GetHandle(int deviceId, int threadId)
{
	std::pair<int, int> key(deviceId, threadId);

	HandleMap::iterator iter = _handleMap.find(key);

	if (iter != _handleMap.end())
		return iter->second;

	CuContext &ret = _handleMap[key];
	ret.Device = deviceId;

	cudaSetDevice(deviceId);

	cublasCreate_v2(&ret.CublasHandle);

	return ret;
}
