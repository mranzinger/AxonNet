/*
 * single_input_layer.h
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */


#pragma once

#include "layer_base.h"

class SingleInputLayer
	: public LayerBase
{
scope_protected:
	std::string _inputName;

scope_public:
	SingleInputLayer() = default;
	SingleInputLayer(std::string name);
	SingleInputLayer(std::string name, std::string inputName);

	virtual void Compute(ParamMap &inputMap, bool isTraining) override final;
	virtual void Backprop(const ParamMap &computeMap, ParamMap &inputErrorMap) override final;

	virtual void SetNet(NeuralNet *net) override;

	const std::string InputName() const { return _inputName; }
	void SetInputName(std::string inputName) { _inputName = std::move(inputName); }

	friend void BindStruct(const axon::serialization::CStructBinder &binder, SingleInputLayer &layer);

#ifdef _UNIT_TESTS_
	Params UTBackprop(const Params &input, const Params &outputErrors)
	{
	    Params output = SCompute(input, true);

	    return SBackprop(input, output, outputErrors);
	}
#endif

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) = 0;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput,
							 const Params &outputErrors) = 0;
};


