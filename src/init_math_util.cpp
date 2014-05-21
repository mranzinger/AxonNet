#include "math_util.h"

#include <random>

using namespace std;

CThreadPool& GetThreadPool()
{
    static CThreadPool *s_threadPool = nullptr;

    if (!s_threadPool)
        s_threadPool = new CThreadPool();

    return *s_threadPool;
}

void InitializeWeights(Vector &vec, Real mean, Real stdDev)
{
	InitializeWeights(vec.data(), vec.data() + vec.size(), mean, stdDev);
}
void InitializeWeights(RMatrix &mat, Real mean, Real stdDev)
{
	InitializeWeights(mat.data(), mat.data() + mat.size(), mean, stdDev);
}

void InitializeWeights(CMatrix& mat, Real mean, Real stdDev)
{
    InitializeWeights(mat.data(), mat.data() + mat.size(), mean, stdDev);
}

void InitializeWeights(Real *iter, Real *end, Real mean, Real stdDev)
{
	//std::default_random_engine engine(1234567);
	//std::random_device engine;

    random_device rd;
    mt19937 engine(rd());

	normal_distribution<Real> dist(mean, stdDev);

	for (Real &val : make_range(iter, end))
	{
		val = dist(engine);
	}
}

void FanInitializeWeights(Vector &vec)
{
	FanInitializeWeights(vec.data(), vec.data() + vec.size());
}

void FanInitializeWeights(RMatrix &mat)
{
	FanInitializeWeights(mat.data(), mat.data() + mat.size(), mat.cols());
}

void FanInitializeWeights(CMatrix& mat)
{
    FanInitializeWeights(mat.data(), mat.data() + mat.size(), mat.cols());
}

void FanInitializeWeights(Real *iter, Real *end, int wtSize)
{
	random_device rd;
    mt19937 engine(rd());

    if (wtSize <= 0)
        wtSize = end - iter;

	float fanIn = 6.0f / sqrt(float(wtSize));

	uniform_real_distribution<Real> dist(-fanIn, fanIn);

	for (Real &val : make_range(iter, end))
		val = dist(engine);
}


