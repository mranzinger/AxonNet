#pragma once

#include <type_traits>

#include "single_input_layer.h"
#include "functions.h"

#include "cu_neuron_layer.cuh"

template<typename Fn>
class NeuronLayer
	: public SingleInputLayer
{
scope_private:
    std::unique_ptr<ICuNeuronLayer> _cuImpl;

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

    virtual void OnInitCudaDevice(int deviceId) override;
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
    if (_cuImpl)
        return _cuImpl->Compute(input, isTraining);

    Params ret(input, ApplyFunction<Fn>(input.GetHostMatrix()));

	return ret;
}

template<typename Fn>
Params NeuronLayer<Fn>::SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
    if (_cuImpl)
        return _cuImpl->Backprop(lastInput, lastOutput, outputErrors);

	CMatrix *c = ApplyDerivative<Fn>(lastInput.GetHostMatrix(), lastOutput.GetHostMatrix());

	c->noalias() = c->cwiseProduct(outputErrors.GetHostMatrix());

	Params ret(lastInput, c);

	return std::move(ret);
}

template<typename Fn>
void NeuronLayer<Fn>::OnInitCudaDevice(int deviceId)
{
    using namespace std;

    CuNeuronType type;

    if (is_same<Fn, LinearFn>::value)
        type = CuNeuronType::Cut_Linear;
    else if (is_same<Fn, LogisticFn>::value)
        type = CuNeuronType::Cut_Logistic;
    else if (is_same<Fn, RectifierFn>::value)
        type = CuNeuronType::Cut_Rectifier;
    else if (is_same<Fn, TanhFn>::value)
        type = CuNeuronType::Cut_Tanh;
    else if (is_same<Fn, RampFn>::value)
        type = CuNeuronType::Cut_Ramp;
    else if (is_same<Fn, SoftPlusFn>::value)
        type = CuNeuronType::Cut_SoftPlus;
    else if (is_same<Fn, HardTanhFn>::value)
        type = CuNeuronType::Cut_HardTanh;
    else
        throw runtime_error("The function type for this neuron layer is not supported "
                            " on cuda currently.");

    _cuImpl.reset(CreateCuNeuronLayer(deviceId, type));
}
