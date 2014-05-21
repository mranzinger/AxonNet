#pragma once

#include <random>

#include "single_input_layer.h"

class DropRand
{
private:
	std::mt19937 _engine;
	std::uniform_real_distribution<Real> _dist;

public:
	DropRand()
		: _dist(0, 1)
    {
        std::random_device rd;
        _engine.seed(rd());
    }

	Real Next() {
		return _dist(_engine);
	}
};

class NEURAL_NET_API DropoutLayer
	: public SingleInputLayer
{
scope_private:
	Real _dropout;
	
	DropRand _rand;

scope_public:
	typedef std::shared_ptr<DropoutLayer> Ptr;

	DropoutLayer(Real dropout = 0.5f) : DropoutLayer("", dropout) { }
	DropoutLayer(std::string name, Real dropout = 0.5f);

	virtual std::string GetLayerType() const override {
		return "Dropout Layer";
	}

	friend void BindStruct(const aser::CStructBinder &binder, DropoutLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
};
