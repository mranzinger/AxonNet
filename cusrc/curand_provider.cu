
#include "curand_provider.cuh"

#include <stdexcept>
#include <sys/time.h>

#include <cuda_runtime.h>

#include "cumath_functions.cuh"

using namespace std;

CURandProvider *CURandProvider::s_instance = NULL;

CURandProvider::CURandProvider()
{
}

CURandProvider::~CURandProvider()
{
}

curandState* CURandProvider::GetRandomStates(int device, uint32_t requiredLen)
{
    return GetInstance()->p_GetRandomStates(device, requiredLen);
}

curandState* CURandProvider::GetRandomStates(int device, dim3 gridDim, dim3 blockDim)
{
    return GetInstance()->p_GetRandomStates(device, gridDim, blockDim);
}

CURandProvider* CURandProvider::GetInstance()
{
    if (s_instance == NULL)
        s_instance = new CURandProvider();

    return s_instance;
}

__global__ void cudaInitRandoms(curandState *state, uint64_t seed, uint32_t maxId)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= maxId)
        return;

    curand_init(seed, id, 0, state + id);
}

curandState* CURandProvider::p_GetRandomStates(int device, uint32_t requiredLen)
{
    CURandDeviceInfo &devInfo = m_buffers[device];

    if (devInfo.ArrayLen >= requiredLen)
        return devInfo.States;

    bool tryPreserve = devInfo.States != NULL;
    curandState *newBuff = NULL;

    cudaError_t err = cudaMalloc(&newBuff, requiredLen * sizeof(curandState));

    if (tryPreserve && err != cudaSuccess)
    {
        // Free the old buffer I suppose. This means that a full re-init will need
        // to be performed
        cudaFree(devInfo.States);
        devInfo.ArrayLen = 0;
        devInfo.States = NULL;

        tryPreserve = false;

        err = cudaMalloc(&newBuff, requiredLen * sizeof(curandState));
    }

    if (err != cudaSuccess)
        throw runtime_error("Unable to allocate random buffer large enough to satisfy request.");

    uint32_t offset = 0;

    if (tryPreserve)
    {
        cudaMemcpy(newBuff, devInfo.States, devInfo.ArrayLen * sizeof(curandState),
                   cudaMemcpyDeviceToDevice);
        cudaFree(devInfo.States);

        offset = devInfo.ArrayLen;
    }

    devInfo.ArrayLen = requiredLen;
    devInfo.States = newBuff;

    // Initialize all of the new random states
    cudaInitRandoms
        <<<round_up(requiredLen - offset, 128), 128>>>
            (devInfo.States + offset,
             time(NULL),
             requiredLen - offset);

    return devInfo.States;
}

curandState* CURandProvider::p_GetRandomStates(int device, dim3 gridDim, dim3 blockDim)
{
    uint32_t reqLen = (gridDim.z * blockDim.z) *
                      (gridDim.y * blockDim.y) *
                      (gridDim.x * blockDim.x);

    return p_GetRandomStates(device, reqLen);
}


