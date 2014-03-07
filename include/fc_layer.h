#pragma once

#include <stdexcept>

#include "layer_base.h"
#include "functions.h"

class FCLayerConfig
	: public LayerConfig
{
public:
	typedef std::shared_ptr<FCLayerConfig> Ptr;

	FCLayerConfig() { }
	FCLayerConfig(std::string name, Matrix weights, Vector biases)
		: LayerConfig(std::move(name)), Weights(std::move(weights)), Biases(std::move(biases))
	{
	}

	Matrix Weights;
	Vector Biases;
};

template<typename Fn>
class FCLayer
	: public LayerBase
{
private:
	Matrix _weights;
	Vector _biases;

	Vector _linOut;

public:
	typedef std::shared_ptr<FCLayer> Ptr;

	FCLayer() { }
	FCLayer(std::string name, size_t numInputs, size_t numOutputs);
	FCLayer(std::string name, Matrix weights, Vector biases);

	virtual std::string GetLayerType() const override {
		return Fn::Type() + " FC Layer";
	}

	virtual Vector Compute(int threadIdx, const Vector &input, bool isTraining) override;
	virtual Vector Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, const Vector &outputErrors) override;

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config) override;
	virtual LayerConfig::Ptr GetConfig() const override;
};

typedef FCLayer<LinearFn> LinearFCLayer;
typedef FCLayer<LogisticFn> LogisticFCLayer;
typedef FCLayer<RectifierFn> RectifierFCLayer;
typedef FCLayer<TanhFn> TanhFCLayer;
typedef FCLayer<RampFn> RampFCLayer;
typedef FCLayer<SoftPlusFn> SoftPlusFCLayer;
typedef FCLayer<HardTanhFn> HardTanhFCLayer;

template<typename Fn>
FCLayer<Fn>::FCLayer(std::string name, size_t numInputs, size_t numOutputs)
	: LayerBase(std::move(name)), _weights(numOutputs, numInputs), _biases(numOutputs)
{
	InitializeWeights(_weights, 0, 5);
	InitializeWeights(_biases, 0, 5);
}

template<typename Fn>
FCLayer<Fn>::FCLayer(std::string name, Matrix weights, Vector biases)
	: LayerBase(std::move(name)), _weights(std::move(weights)), _biases(std::move(biases))
{

}

template<typename Fn>
LayerConfig::Ptr FCLayer<Fn>::GetConfig() const
{
	return std::make_shared<FCLayerConfig>(_name, _weights, _biases);
}

template<typename Fn>
void FCLayer<Fn>::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	auto fcConfig = std::dynamic_pointer_cast<FCLayerConfig>(config);

	if (!fcConfig)
		throw std::runtime_error("The specified layer config is invalid. It must be of type FCLayerConfig.");

	LayerBase::InitializeFromConfig(config);

	_weights = fcConfig->Weights;
	_biases = fcConfig->Biases;
}

template<typename Fn>
Vector FCLayer<Fn>::Compute(int threadIdx, const Vector &input, bool isTraining)
{
	Vector lin = _weights * input + _biases;

	if (isTraining)
		_linOut = lin;

	Vector op = lin.unaryExpr([](Real val) { return Fn::Compute(val); });

	return op;
}

template<typename Fn>
Vector FCLayer<Fn>::Backprop(int threadIdx, const Vector &lastInput, const Vector &lastOutput, 
	const Vector &outputErrors)
{
	Vector inputErrors = _weights.transpose() * outputErrors;

	auto dv = _linOut.unaryExpr(
		[](Real val) 
		{ 
			return Fn::Derivative(val); 
		}
	);

	Vector errLayer = outputErrors.binaryExpr(dv, 
		[](Real a, Real b) 
		{ 
			return a * b; 
	});

	_weights.noalias() += _learningRate * (errLayer * lastInput.transpose());
	_biases.noalias() += errLayer;

	return inputErrors;
}