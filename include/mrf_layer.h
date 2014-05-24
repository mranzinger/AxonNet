/*
 * mrf_layer.h
 *
 *  Created on: May 23, 2014
 *      Author: mike
 */

#pragma once

#include "single_input_layer.h"

class NEURAL_NET_API MRFLayer
	: public SingleInputLayer
{
scope_private:
	int _width, _height;

scope_public:
	MRFLayer() = default;
	MRFLayer(std::string name,
			 size_t width, size_t height);

	virtual std::string GetLayerType() const override {
		return "Max Response Field Layer";
	}

	friend void BindStruct(const aser::CStructBinder &binder, MRFLayer &layer);

scope_protected:
	virtual Params SCompute(const Params &input, bool isTraining) override;
	virtual Params SBackprop(const Params &lastInput, const Params &lastOutput, const Params &outputErrors) override;
};
