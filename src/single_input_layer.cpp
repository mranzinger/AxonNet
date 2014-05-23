/*
 * single_input_layer.cpp
 *
 *  Created on: May 20, 2014
 *      Author: mike
 */

#include "single_input_layer.h"

#include "neural_net.h"
#include "i_train_provider.h"

using namespace std;

SingleInputLayer::SingleInputLayer(std::string name)
	: SingleInputLayer(move(name), "")
{
}

SingleInputLayer::SingleInputLayer(std::string name, std::string inputName)
	: LayerBase(move(name)), _inputName(move(inputName))
{
}

void SingleInputLayer::Compute(ParamMap& inputMap, bool isTraining)
{
	const Params &input = *GetData(inputMap, _inputName);

	Params output = SCompute(input, isTraining);

	inputMap[_name] = move(output);
}

void SingleInputLayer::Backprop(const ParamMap& computeMap,
		ParamMap& inputErrorMap)
{
	const Params &lastInput = *GetData(computeMap, _inputName);
	const Params &lastOutput = *GetData(computeMap, _name);

	const Params &outputErrors = *GetData(inputErrorMap, _name);

	Params inputErrors = SBackprop(lastInput, lastOutput, outputErrors);

	inputErrorMap[_inputName] = move(inputErrors);
}

void SingleInputLayer::SetNet(NeuralNet* net)
{
	LayerBase::SetNet(net);

	if (_inputName.empty())
	{
		int myIndex = net->GetLayerIndex(this);

		// If this layer is the first, use the default input name
		if (myIndex == 0)
			_inputName = ITrainProvider::DEFAULT_INPUT_NAME;
		else if (myIndex > 0)
		{
			// Use the previous layers name as the input name.
			// This creates a linear chain type network, which is the most common
			ILayer::Ptr prevLayer = net->GetLayer(myIndex - 1);

			_inputName = prevLayer->GetLayerName();
		}
		else
			throw runtime_error("This layer doesn't even belong to the network.");
	}
}

void BindStruct(const axon::serialization::CStructBinder &binder, SingleInputLayer &layer)
{
	BindStruct(binder, (LayerBase&)layer);

	binder("inputName", layer._inputName);
}
