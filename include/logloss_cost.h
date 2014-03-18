#pragma once

#include "i_cost.h"

class NEURAL_NET_API LogLossCost
	: public ICost
{
public:
	typedef std::shared_ptr<LogLossCost> Ptr;

	virtual std::string GetType() const {
		return "Log Loss";
	}

	virtual Real Compute(const Params &pred, const Params &labels) override;
	virtual Params ComputeGrad(const Params &pred, const Params &labels) override;
};