#pragma once

#include "layer_base.h"
#include "functions.h"

template<typename Fn>
class NeuronLayer
	: public LayerBase
{
public:
	typedef std::shared_ptr<NeuronLayer> Ptr;

	NeuronLayer() { }
	NeuronLayer(std::string name) : LayerBase(std::move(name)) { }

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
	return ApplyFunction<Fn>(input);
}

template<typename Fn>
Vector NeuronLayer<Fn>::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors)
{
	Vector v = ApplyDerivative<Fn>(lastInput, lastOutput);

	v.noalias() = v.cwiseProduct(outputErrors);

	return v;
}