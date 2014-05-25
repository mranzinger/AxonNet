/*
 * mrf_layer.h
 *
 *  Created on: May 23, 2014
 *      Author: mike
 */

#pragma once

#include <vector>

#include "single_input_layer.h"
#include "util/enum_to_string.h"

enum class MRFFunction
{
	Default,
	Abs,
	Squared,
	SignSquared
};

ENUM_IO_FWD(MRFFunction, );

class NEURAL_NET_API MRFLayer
	: public LayerBase
{
scope_private:
	int _width, _height;
	MRFFunction _function;

	std::string _inputName;

scope_public:
	MRFLayer() = default;
	MRFLayer(std::string name,
			 size_t width, size_t height);
	MRFLayer(std::string name, std::string inputName,
			 size_t width, size_t height);

	virtual std::string GetLayerType() const override {
		return "Max Response Field Layer";
	}

    virtual void Compute(ParamMap &inputMap, bool isTraining) override;
	virtual void Backprop(const ParamMap &computeMap, ParamMap &inputErrorMap) override;

	friend void BindStruct(const aser::CStructBinder &binder, MRFLayer &layer);

scope_private:
	void CalcSAT(const RMap &inputField, RMatrix &sumAreaTable, int depth) const;

	const std::string &GetInputName();
};
