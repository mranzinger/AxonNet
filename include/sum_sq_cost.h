#pragma once

#include "i_cost.h"

class NEURAL_NET_API SumSqCost
	: public ICost
{
public:
	typedef std::shared_ptr<SumSqCost> Ptr;

	virtual std::string GetType() const {
		return "Sum Squared Loss";
	}

	virtual Real Compute(const Params &pred, const Params &labels) override;
	virtual Params ComputeGrad(const Params &pred, const Params &labels) override;
};

