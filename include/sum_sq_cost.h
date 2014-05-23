#pragma once

#include "simple_cost.h"

class NEURAL_NET_API SumSqCost
	: public SimpleCost
{
scope_public:
	typedef std::shared_ptr<SumSqCost> Ptr;

	SumSqCost();
	SumSqCost(std::string inputName);
	SumSqCost(std::string inputName, std::string labelName);

	virtual std::string GetType() const {
		return "Sum Squared Loss";
	}

	friend void BindStruct(const aser::CStructBinder &binder, SumSqCost &cost);

scope_protected:
	virtual CostMap SCompute(const Params &pred, const Params &labels) override;
	virtual Params SComputeGrad(const Params &pred, const Params &labels) override;
};

