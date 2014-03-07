#include "math_util.h"

#include <random>

using namespace std;

void InitializeWeights(Vector &vec, Real mean, Real stdDev)
{
	InitializeWeights(vec.data(), vec.data() + vec.size(), mean, stdDev);
}
void InitializeWeights(Matrix &mat, Real mean, Real stdDev)
{
	InitializeWeights(mat.data(), mat.data() + mat.size(), mean, stdDev);
}
void InitializeWeights(Real *iter, Real *end, Real mean, Real stdDev)
{
	//std::default_random_engine engine(1234567);
	std::random_device engine;

	std::normal_distribution<Real> dist(mean, stdDev);

	for (Real &val : make_range(iter, end))
	{
		val = dist(engine);
	}
}