/*
 * File description: simple_cost.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "simple_cost.h"

using namespace std;

SimpleCost::SimpleCost(std::string inputName)
    : _inputName(move(inputName))
{
}

SimpleCost::SimpleCost(std::string inputName, std::string labelName)
    : _inputName(move(inputName)), _labelName(move(labelName))
{
}

Real SimpleCost::Compute(const ParamMap& inputs)
{
    const Params &input = *FindParams(inputs, _inputName);
    const Params &labels = *FindParams(inputs,
                                       _labelName.empty() ?
                                              ITrainProvider::DEFAULT_LABEL_NAME
                                          :   _labelName);

    return SCompute(input, labels);
}

void SimpleCost::ComputeGrad(const ParamMap& inputs, ParamMap& inputErrors)
{
    const Params &input = *FindParams(inputs, _inputName);
    const Params &labels = *FindParams(inputs,
                                       _labelName.empty() ?
                                              ITrainProvider::DEFAULT_LABEL_NAME
                                          :   _labelName);

    Params ipCost = SComputeGrad(input, labels);

    inputErrors[_inputName] = move(ipCost);
}


void BindStruct(const aser::CStructBinder &binder, SimpleCost &cost)
{
    binder("inputName", cost._inputName)
          ("labelName", cost._labelName);
}
