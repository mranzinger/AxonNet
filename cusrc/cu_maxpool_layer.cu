/*
 * cu_maxpool_layer.cu
 *
 *  Created on: Jun 9, 2014
 *      Author: mike
 */

#include "cu_maxpool_layer.cuh"

#include "cumat.cuh"
#include "cumath_functions.cuh"
#include "cusetup_provider.cuh"

using namespace std;

CuMaxPoolLayer::CuMaxPoolLayer(int deviceId)
	: _cacheCompute(NULL), _cacheBackprop(NULL)
{
	SetWindowSize(0, 0);
	ResetStepSize();

	InitDevice(deviceId);
}

CuMaxPoolLayer::CuMaxPoolLayer(int deviceId, uint32_t windowSizeX, uint32_t windowSizeY)
	: _cacheCompute(NULL), _cacheBackprop(NULL)
{
	SetWindowSize(windowSizeX, windowSizeY);
	ResetStepSize();

	InitDevice(deviceId);
}

CuMaxPoolLayer::CuMaxPoolLayer(int deviceId, uint32_t windowSizeX, uint32_t windowSizeY,
		uint32_t stepX, uint32_t stepY)
	: _cacheCompute(NULL), _cacheBackprop(NULL)
{
	SetWindowSize(windowSizeX, windowSizeY);
	SetStepSize(stepX, stepY);

	InitDevice(deviceId);
}

CuMaxPoolLayer::~CuMaxPoolLayer()
{
	delete _cacheCompute;
	delete _cacheBackprop;
}

void CuMaxPoolLayer::SetWindowSize(uint32_t windowSizeX, uint32_t windowSizeY)
{
	_windowSizeX = windowSizeX;
	_windowSizeY = windowSizeY;
}

void CuMaxPoolLayer::SetStepSize(uint32_t stepX, uint32_t stepY)
{
	_stepX = stepX;
	_stepY = stepY;
}

void CuMaxPoolLayer::ResetStepSize()
{
	_stepX = _windowSizeX;
	_stepY = _windowSizeY;
}

void CuMaxPoolLayer::InitDevice(int deviceId)
{
	_handle = CuSetupProvider::GetHandle(deviceId);

	delete _cacheCompute;
	delete _cacheBackprop;

	_cacheCompute = new CuMat(_handle);
	_cacheCompute->SetSharedModify(true);
	_cacheBackprop = new CuMat(_handle);
	_cacheBackprop->SetSharedModify(true);
}

void CuMaxPoolLayer::EnsureStep()
{
	if (!_stepX || !_stepY)
		ResetStepSize();
}

__global__ void CuMaxPoolLayer_Compute(const Real *pFullInput, Real *pFullOutput,
									   uint32_t ipWidth, uint32_t ipHeight,
									   uint32_t opWidth, uint32_t opHeight,
									   uint32_t depth,
									   uint32_t windowSizeX, uint32_t windowSizeY,
									   uint32_t stepX, uint32_t stepY)
{
	uint32_t layer = blockIdx.z * blockDim.z + threadIdx.z;

	const Real *pInput = pFullInput + layer * (ipWidth * ipHeight * depth);
	Real *pOutput = pFullOutput + layer * (opWidth * opHeight * depth);

	uint32_t blockX = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t opX = blockX / depth;

	if (opX >= opWidth)
		return;

	uint32_t opY = blockIdx.y * blockDim.y + threadIdx.y;

	if (opY >= opHeight)
		return;

	uint32_t dOff = blockX % depth;

	uint32_t ipX = opX * stepX;
	uint32_t ipY = opY * stepY;

	uint32_t xEnd = min(ipX + windowSizeX, ipWidth);
	uint32_t yEnd = min(ipY + windowSizeY, ipHeight);

	Real max = -1000000000.0f;

	for (uint32_t y = ipY; y < yEnd; ++y)
	{
		for (uint32_t x = ipX; x < xEnd; ++x)
		{
			Real val = pInput[y * ipWidth * depth + x * depth + dOff];

			if (val > max)
				max = val;
		}
	}

	pOutput[opY * opWidth * depth + blockX] = max;
}

__global__ void CuMaxPoolLayer_Backprop(const Real *pFullInput, const Real *pFullOutput,
										const Real *pFullOutputErrors,
										Real *pFullInputErrors,
										uint32_t ipWidth, uint32_t ipHeight,
										uint32_t opWidth, uint32_t opHeight,
										uint32_t depth,
										uint32_t windowSizeX, uint32_t windowSizeY,
										uint32_t stepX, uint32_t stepY)
{
	uint32_t layer = blockIdx.z * blockDim.z + threadIdx.z;

	const Real *pInput = pFullInput + layer * (ipWidth * ipHeight * depth);
	const Real *pOutput = pFullOutput + layer * (opWidth * opHeight * depth);
	const Real *pOutputErrors = pFullOutputErrors + layer * (opWidth * opHeight * depth);
	Real *pInputErrors = pFullInputErrors + layer * (ipWidth * ipHeight * depth);

	uint32_t blockX = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t opX = blockX / depth;

	if (opX >= opWidth)
		return;

	uint32_t opY = blockIdx.y * blockDim.y + threadIdx.y;

	if (opY >= opHeight)
		return;

	uint32_t dOff = blockX % depth;

	uint32_t ipX = opX * stepX;
	uint32_t ipY = opY * stepY;

	uint32_t xEnd = min(ipX + windowSizeX, ipWidth);
	uint32_t yEnd = min(ipY + windowSizeY, ipHeight);

	const uint32_t opIdx = opY * opWidth * depth + blockX;

	const Real opVal = pOutput[opIdx];
	const Real opErrVal = pOutputErrors[opIdx];

	for (uint32_t y = ipY; y < yEnd; ++y)
	{
		for (uint32_t x = ipX; x < xEnd; ++x)
		{
			const uint32_t ipIdx = y * ipWidth * depth + x * depth + dOff;

			Real ipVal = pInput[ipIdx];

			Real &ipErrVal = pInputErrors[ipIdx];

			if (ipVal == opVal)
			{
				ipErrVal = opErrVal;
			}
			else
				ipErrVal = 0.0f;
		}
	}
}

Params CuMaxPoolLayer::Compute(const Params& input)
{
	EnsureStep();

	if (!_windowSizeX || !_windowSizeY)
		return input;

	const uint32_t ipWidth = input.Width;
	const uint32_t ipHeight = input.Height;
	const uint32_t depth = input.Depth;

	const uint32_t opWidth = (uint32_t) ceil(ipWidth / float(_stepX));
	const uint32_t opHeight = (uint32_t) ceil(ipHeight / float(_stepY));



	dim3 threads(32, 32, 1);
	dim3 blocks = round_up(opWidth * depth, opHeight, input.Cols, threads);

	const CuMat &mInput = input.GetCudaMatrix(_handle);

	_cacheCompute->ResizeLike(mInput);
	Params output(opWidth, opHeight, depth, new CuMat(*_cacheCompute));

	CuMat &mOutput = output.GetCudaMatrix(_handle);

	CuMaxPoolLayer_Compute
#ifdef _CUDA_COMPILE_
		<<<blocks, threads>>>
#endif
		(mInput.Buff(), mOutput.Buff(),
		 ipWidth, ipHeight,
		 opWidth, opHeight,
		 depth,
		 _windowSizeX, _windowSizeY,
		 _stepX, _stepY);

	return output;
}

Params CuMaxPoolLayer::Backprop(const Params& input, const Params& lastOutput,
		const Params& outputErrors)
{
	EnsureStep();

	if (!_windowSizeX || !_windowSizeY)
		return outputErrors;



	const CuMat &mInput = input.GetCudaMatrix(_handle);
	const CuMat &mOutput = lastOutput.GetCudaMatrix(_handle);
	const CuMat &mOutputErrors = outputErrors.GetCudaMatrix(_handle);

	_cacheBackprop->ResizeLike(mInput);
	Params inputErrors(input, new CuMat(*_cacheBackprop));

	CuMat &mInputErrors = inputErrors.GetCudaMatrix(_handle);

	const uint32_t ipWidth = input.Width;
	const uint32_t ipHeight = input.Height;
	const uint32_t depth = input.Depth;

	const uint32_t opWidth = lastOutput.Width;
	const uint32_t opHeight = lastOutput.Height;

	dim3 threads(32, 32, 1);
	dim3 blocks = round_up(opWidth * depth, opHeight, input.Cols, threads);

	CuMaxPoolLayer_Backprop
#ifdef _CUDA_COMPILE_
		<<<blocks, threads>>>
#endif
		(mInput.Buff(), mOutput.Buff(), mOutputErrors.Buff(),
		 mInputErrors.Buff(),
		 ipWidth, ipHeight,
		 opWidth, opHeight,
		 depth,
		 _windowSizeX, _windowSizeY,
		 _stepX, _stepY);

	return inputErrors;
}


