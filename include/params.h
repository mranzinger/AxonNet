#pragma once

#include <map>
#include <string>

#include "math_util.h"

class CuMat;
class Params;

void swap(Params &a, Params &b);

class NEURAL_NET_API Params
{
public:
	uint32_t Width;
	uint32_t Height;
	uint32_t Depth;

	uint32_t Rows;
	uint32_t Cols;

	Params();
	explicit Params(CMatrix *hostMat);
	explicit Params(CuMat *cudaMat);
	Params(size_t width, size_t height, size_t depth, CMatrix *hostMat);
	Params(size_t width, size_t height, size_t depth, CuMat *cudaMat);
	Params(const Params &other);
	Params(const Params &like, CMatrix *hostMat);
	Params(const Params &like, CuMat *cudaMat);

#ifndef _CUDA_COMPILE_
	Params(Params &&other);
#endif

	~Params();

	bool IsOnHost() const { return _hostMat != NULL; }
	bool IsOnDevice() const { return _cudaMat != NULL; }

	const CMatrix &GetHostMatrix() const;
	CMatrix &GetHostMatrix();
	const CuMat &GetCudaMatrix(CuContext handle) const;
	CuMat &GetCudaMatrix(CuContext handle);

	Params &operator=(Params other);

	friend void swap(Params &a, Params &b);

private:
	uint32_t *_refCt;
	mutable CMatrix *_hostMat;
	mutable CuMat *_cudaMat;
};

typedef std::vector<Params> MultiParams;
typedef std::map<std::string, Params> ParamMap;

