#pragma once

#include "i_cost.h"

class NEURAL_NET_API LogLossCost
	: public ICost
{
private:
	NeuralNet *_net;
	bool _checked;
	bool _outputIsSoftmax;

public:
	typedef std::shared_ptr<LogLossCost> Ptr;

	LogLossCost() : _net(nullptr), _checked(false), _outputIsSoftmax(false) { }

	virtual std::string GetType() const {
		return "Log Loss";
	}

	virtual Real Compute(const Params &pred, const Params &labels) override;
	virtual Params ComputeGrad(const Params &pred, const Params &labels) override;

	virtual void SetNet(NeuralNet *net) override { _net = net; }

private:
	void EstablishContext();
};
