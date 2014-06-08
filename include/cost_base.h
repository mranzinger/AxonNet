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

    IDevicePreference::Ptr _devicePref;

scope_public:
    CostBase();

    virtual void SetNet(NeuralNet *net) override;

    virtual void SetDevicePreference(IDevicePreference::Ptr pref) override;

    friend void BindStruct(const aser::CStructBinder &binder, CostBase &cost);

scope_protected:
    const Params *FindParams(const ParamMap &input, const std::string &name, bool enforce = true);
    Params *FindParams(ParamMap &input, const std::string &name, bool enforce = true);

    virtual void OnInitialized();
	virtual void OnInitCPUDevice() { }
	virtual void OnInitCudaDevice(int deviceId) { }
};


