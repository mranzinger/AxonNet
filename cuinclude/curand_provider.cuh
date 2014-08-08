/*
 * curand_provider.cuh
 *
 *  Created on: Aug 7, 2014
 *      Author: mike
 */

#pragma once

#include <map>
#include <stdint.h>

#include <curand_kernel.h>

struct CURandDeviceInfo
{
    curandState *States;
    uint64_t ArrayLen;

    CURandDeviceInfo() : States(NULL), ArrayLen(0) { }
};

class CURandProvider
{
public:
    ~CURandProvider();

    static curandState *GetRandomStates(int device, uint32_t requiredLen);
    static curandState *GetRandomStates(int device, dim3 gridDim, dim3 blockDim);

private:
    static CURandProvider *GetInstance();

    curandState *p_GetRandomStates(int device, uint32_t requiredLen);
    curandState *p_GetRandomStates(int device, dim3 gridDim, dim3 blockDim);

    CURandProvider();
    CURandProvider(const CURandProvider &); // Not implemented on purpose

    static CURandProvider *s_instance;

    std::map<int, CURandDeviceInfo> m_buffers;
};


