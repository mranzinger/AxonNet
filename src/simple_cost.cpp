/*
 * File description: simple_cost.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "simple_cost.h"

#include "i_train_provider.h"
#include "neural_net.h"

using namespace std;

SimpleCost::SimpleCost(std::string inputName)
    : _inputName(move(inputName))
{
}

SimpleCost::SimpleCost(std::string inputName, std::string labelName)
    : _inputName(move(inputName)), _labelName(move(labelName))
{
}

CostMap SimpleCost::Compute(const ParamMap& inputs)
{
	const std::string &inputName = !_inputName.empty() ?
                                        _inputName
                                    :   _net->GetLayer(_net->NumLayers() - 1)->GetLayerName();

    const Params &input = *FindParams(inputs, inputName);
    const Params &labels = *FindParams(inputs,
                                       _labelName.empty() ?
                                              ITrainProvider::DEFAULT_LABEL_NAME
                                          :   _labelName);

    return SCompute(input, labels);
}

void SimpleCost::ComputeGrad(const ParamMap& inputs, ParamMap& inputErrors)
{
    const std::string &inputName = !_inputName.empty() ?
                                        _inputName
                                    :   _net->GetLayer(_net->NumLayers() - 1)->GetLayerName();

    const Params *input = FindParams(inputs, inputName);

    const Params &labels = *FindParams(inputs,
                                       _labelName.empty() ?
                                              ITrainProvider::DEFAULT_LABEL_NAME
                                          :   _labelName);

    Params ipCost = SComputeGrad(*input, labels);

    inputErrors[inputName] = move(ipCost);
}

bool SimpleCost::IsBetter(const CostMap& a, const CostMap& b) const
{
    if (a.empty())
        return false;

    auto aIter = a.find(CostMap::PRIMARY_NAME);
    auto bIter = b.find(CostMap::PRIMARY_NAME);

    if (aIter == a.end())
        return false;
    if (bIter == b.end())
        return true;

    return aIter->second < bIter->second;
}

void BindStruct(const aser::CStructBinder &binder, SimpleCost &cost)
{
    binder("inputName", cost._inputName)
          ("labelName", cost._labelName);
}

