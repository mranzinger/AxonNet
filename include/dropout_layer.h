#pragma once

#include <random>

#include "layer_base.h"

class DropRand
{
private:
	std::default_random_engine _engine;
	std::uniform_int_distribution<uint64_t> _dist;

public:
	DropRand()
		: _dist(0, std::numeric_limits<uint64_t>::max()) { }

	uint64_t Next() {
		return _dist(_engine);
	}
};

class NEURAL_NET_API DropoutLayer
	: public LayerBase
{
	typedef std::vector<uint64_t> RandVec;
	typedef std::vector<RandVec> RandVecs;
	typedef std::vector<DropRand> Gens;

private:
	Real _dropout;
	
	Gens _trainGens;
	RandVecs _trainRands;

public:
	typedef std::shared_ptr<DropoutLayer> Ptr;

	DropoutLayer(Real dropout = 0.5f) : DropoutLayer("", dropout) { }
	DropoutLayer(std::string name, Real dropout = 0.5f);

	virtual std::string GetLayerType() const override {
		return "Dropout Layer";
	}

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;

	virtual void PrepareForThreads(size_t num) override;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, DropoutLayer &layer);

private:
	void Dropout(int threadIdx, const Vector &input, Vector &output);
};