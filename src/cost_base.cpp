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
    _devicePref = CPUDevicePreference::Instance;
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

void CostBase::SetDevicePreference(IDevicePreference::Ptr pref)
{
	_devicePref = move(pref);

	OnInitialized();
}

void CostBase::OnInitialized()
{
	assert(_devicePref);

    switch (_devicePref->Type())
    {
    default:
        throw runtime_error("Unsupported device preference.");
    case DevicePreference::CPU:
        OnInitCPUDevice();
        break;
    case DevicePreference::Cuda:
        OnInitCudaDevice(dynamic_cast<CudaDevicePreference*>(_devicePref.get())->DeviceId);
        break;
    }
}

void BindStruct(const aser::CStructBinder &binder, CostBase &cost)
{
	binder("device", cost._devicePref);

	if (binder.IsRead())
	{
		cost.OnInitialized();
	}
}
