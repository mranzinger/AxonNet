#pragma once

#include "simple_cost.h"

class NEURAL_NET_API LogLossCost
	: public SimpleCost
{
scope_private:
	bool _checked;
	bool _outputIsSoftmax;

scope_public:
	typedef std::shared_ptr<LogLossCost> Ptr;

	LogLossCost();
	LogLossCost(std::string inputName);
	LogLossCost(std::string inputName, std::string labelName);

	virtual std::string GetType() const {
		return "Log Loss";
	}

	friend void BindStruct(const aser::CStructBinder &binder, LogLossCost &cost);

scope_protected:
    virtual CostMap SCompute(const Params &pred, const Params &labels) override;
    virtual Params SComputeGrad(const Params &pred, const Params &labels) override;

scope_private:
	void EstablishContext();
};
