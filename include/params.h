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

	/*
	 * Input data matrix. Supports mini-batch when the number of
	 * columns > 1. Data is stored column major, so accessing the kth element
	 * of the ith column is (i * #rows) + k
	 */
	CMatrix *HostMat;

	CuMat *CudaMat;

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

	Params &operator=(Params other);

	friend void swap(Params &a, Params &b);

private:
	uint32_t *_refCt;
};

typedef std::vector<Params> MultiParams;
typedef std::map<std::string, Params> ParamMap;

