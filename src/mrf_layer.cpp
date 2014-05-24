/*
 * mrf_layer.cpp
 *
 *  Created on: May 23, 2014
 *      Author: mike
 */

#include "mrf_layer.h"

using namespace std;

MRFLayer::MRFLayer(std::string name,
				   size_t width, size_t height)
	: SingleInputLayer(move(name)),
	  _width(width), _height(height)
{
}

Params MRFLayer::SCompute(const Params& input, bool isTraining)
{
}

Params MRFLayer::SBackprop(const Params& lastInput, const Params& lastOutput,
		const Params& outputErrors)
{
}

void BindStruct(const aser::CStructBinder &binder, MRFLayer &layer)
{
	binder("width", layer._width)
		  ("height", layer._height);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MRFLayer, MRFLayer);
