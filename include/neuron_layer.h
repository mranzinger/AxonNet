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

	virtual Params Compute(int threadIdx, const Params &input, bool isTraining) override;
	virtual Params Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
};

typedef NeuronLayer<LinearFn> LinearNeuronLayer;
typedef NeuronLayer<LogisticFn> LogisticNeuronLayer;
typedef NeuronLayer<RectifierFn> RectifierNeuronLayer;
typedef NeuronLayer<TanhFn> TanhNeuronLayer;
typedef NeuronLayer<RampFn> RampNeuronLayer;
typedef NeuronLayer<SoftPlusFn> SoftPlusNeuronLayer;
typedef NeuronLayer<HardTanhFn> HardTanhNeuronLayer;

template<typename Fn>
Params NeuronLayer<Fn>::Compute(int threadIdx, const Params &input, bool isTraining)
{
	return Params(input, ApplyFunction<Fn>(input.Data));
}

template<typename Fn>
Params NeuronLayer<Fn>::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params ret(lastInput, CMatrix());

	ret.Data = ApplyDerivative<Fn>(lastInput.Data, lastOutput.Data);

	ret.Data.noalias() = ret.Data.cwiseProduct(outputErrors.Data);

	return std::move(ret);
}
