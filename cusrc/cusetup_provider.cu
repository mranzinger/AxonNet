/*
 * cusetup_provider.cu
 *
 *  Created on: Jun 4, 2014
 *      Author: mike
 */

#include "cusetup_provider.cuh"

#include <iostream>

using namespace std;

CuSetupProvider::CuSetupProvider()
{
}

CuSetupProvider::~CuSetupProvider()
{
	/*for (HandleMap::iterator iter = _handleMap.begin(),
			                 end = _handleMap.end();
			iter != end;
			++iter)
	{
	    cudaError_t err = cudaSetDevice(iter->second.Device);
	    if (err != cudaSuccess)
	    {
	        cerr << "Failed to set the device before freeing. Error: " << err << endl;
	        exit(1);
	    }

	    err = cudaDeviceSynchronize();
	    if (err != cudaSuccess)
	    {
	        cerr << "Failed to synchronize with the device before freeing the cublas handle. Error: " << err << endl;
	        exit(1);
	    }

		cublasStatus_t status = cublasDestroy_v2(iter->second.CublasHandle);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
		    cerr << "Failed to destroy the cublas handle. Error: " << status << endl;
		}
	}*/
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
