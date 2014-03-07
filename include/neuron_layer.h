#pragma once

#include "layer_base.h"
#include "functions.h"

template<typename Fn>
class NEURAL_NET_API NeuronLayer
	: public LayerBase
{
public:
	typedef std::shared_ptr<NeuronLayer> Ptr;

	virtual std::string GetLayerType() const override {
		return Fn::Type() + " Neuron Layer";
	}

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;
};

typedef NeuronLayer<LinearFn> LinearNeuronLayer;
typedef NeuronLayer<LogisticFn> LogisticNeuronLayer;
typedef NeuronLayer<RectifierFn> RectifierNeuronLayer;
typedef NeuronLayer<TanhFn> TanhNeuronLayer;
typedef NeuronLayer<RampFn> RampNeuronLayer;
typedef NeuronLayer<SoftPlusFn> SoftPlusNeuronLayer;
typedef NeuronLayer<HardTanhFn> HardTanhNeuronLayer;

template<typename Fn>
Vector NeuronLayer<Fn>::Compute(int threadIdx, const Vector &input, bool isTraining)
{
	return ApplyFunction(input);
}

template<typename Fn>
Vector NeuronLayer<Fn>::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors)
{
	return ApplyDerivative(outputErrors);
}