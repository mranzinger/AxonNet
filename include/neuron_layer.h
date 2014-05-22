#pragma once

#include "single_input_layer.h"
#include "functions.h"

template<typename Fn>
class NeuronLayer
	: public SingleInputLayer
{
scope_public:
	typedef std::shared_ptr<NeuronLayer> Ptr;

	NeuronLayer() { }
	NeuronLayer(std::string name)
	    : SingleInputLayer(std::move(name))
	{
	}
	NeuronLayer(std::string name, std::string inputName)
	    : SingleInputLayer(std::move(name), std::move(inputName))
	{
	}

	virtual std::string GetLayerType() const override {
		return Fn::Type() + " Neuron Layer";
	}

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
    virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
};

typedef NeuronLayer<LinearFn> LinearNeuronLayer;
typedef NeuronLayer<LogisticFn> LogisticNeuronLayer;
typedef NeuronLayer<RectifierFn> RectifierNeuronLayer;
typedef NeuronLayer<TanhFn> TanhNeuronLayer;
typedef NeuronLayer<RampFn> RampNeuronLayer;
typedef NeuronLayer<SoftPlusFn> SoftPlusNeuronLayer;
typedef NeuronLayer<HardTanhFn> HardTanhNeuronLayer;

template<typename Fn>
Params NeuronLayer<Fn>::SCompute(const Params &input, bool isTraining)
{
	return Params(input, ApplyFunction<Fn>(input.Data));
}

template<typename Fn>
Params NeuronLayer<Fn>::SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	Params ret(lastInput, CMatrix());

	ret.Data = ApplyDerivative<Fn>(lastInput.Data, lastOutput.Data);

	ret.Data.noalias() = ret.Data.cwiseProduct(outputErrors.Data);

	return std::move(ret);
}
