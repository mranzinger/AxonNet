#include "logloss_cost.h"

using namespace std;

static const Real s_epss = 0.0001;

Real LogLossCost::Compute(const Vector &pred, const Vector &labels)
{
	// Pull the prediction away from 0 and 1
	auto safe = pred.unaryExpr(
		[](Real val)
		{
			return min(max(val, s_epss), 1 - s_epss);
		});

	Real ret = -(labels.binaryExpr(safe,
		[](Real label, Real pred)
		{
			return label * log(pred) + (1 - label) * log(1 - pred);
		}))
			.mean();

	return ret;
}

Vector LogLossCost::ComputeGrad(const Vector &pred, const Vector &labels)
{
	auto safe = pred.unaryExpr(
		[](Real val)
		{
			return min(max(val, s_epss), 1 - s_epss);
		});

	Vector ret = (-1.0 / labels.size()) *
		(labels.binaryExpr(safe,
			[](Real label, Real pred)
			{
				return (label / pred) - ((1 - label) / (1 - pred));
			}));

	return ret;
}

void BindStruct(const axon::serialization::CStructBinder &, LogLossCost&) { }

AXON_SERIALIZE_DERIVED_TYPE(ICost, LogLossCost, LogLossCost);