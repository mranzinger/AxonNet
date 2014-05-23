/*
 * File description: simple_cost.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "cost_base.h"

class NEURAL_NET_API SimpleCost
    : public CostBase
{
scope_protected:
    std::string _inputName;
    std::string _labelName;

scope_public:
    SimpleCost() = default;
    SimpleCost(std::string inputName);
    SimpleCost(std::string inputName, std::string labelName);

    virtual CostMap Compute(const ParamMap &inputs) override final;
    virtual void ComputeGrad(const ParamMap &inputs, ParamMap &inputErrors) override final;

    virtual bool IsBetter(const CostMap &a, const CostMap &b) const override;

    friend void BindStruct(const aser::CStructBinder &binder, SimpleCost &cost);

scope_protected:
    virtual CostMap SCompute(const Params &input, const Params &labels) = 0;
    virtual Params SComputeGrad(const Params &input, const Params &labels) = 0;
};


