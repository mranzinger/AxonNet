/*
 * File description: cost_base.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "i_cost.h"

class NEURAL_NET_API CostBase
    : public virtual ICost
{
scope_protected:
    NeuralNet *_net;

scope_public:
    CostBase();

    virtual void SetNet(NeuralNet *net) override;

scope_protected:
    const Params *FindParams(const ParamMap &input, const std::string &name, bool enforce = true);
    Params *FindParams(ParamMap &input, const std::string &name, bool enforce = true);
};


