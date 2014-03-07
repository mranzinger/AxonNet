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

	virtual Real Compute(const Vector &pred, const Vector &labels) override;
	virtual Vector ComputeGrad(const Vector &pred, const Vector &labels) override;
};