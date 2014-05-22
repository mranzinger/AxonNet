/*
 * File description: cost_base.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cost_base.h"

#include <stdexcept>

using namespace std;

CostBase::CostBase()
    : _net(nullptr)
{
}

void CostBase::SetNet(NeuralNet* net)
{
    _net = net;
}

const Params* CostBase::FindParams(const ParamMap &input,
        const std::string& name, bool enforce)
{
    return FindParams(const_cast<ParamMap&>(input), name, enforce);
}

Params* CostBase::FindParams(ParamMap &input, const std::string& name,
        bool enforce)
{
    auto iter = input.find(name);

    if (iter != input.end())
        return &iter->second;

    if (enforce)
        throw runtime_error("The parameters '" + name + "' were not stored in the parameter map.");

    return nullptr;
}
