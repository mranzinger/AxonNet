/*
 * File description: device_preference.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "device_preference.h"

CPUDevicePreference::Ptr CPUDevicePreference::Instance(new CPUDevicePreference);

AXON_SERIALIZE_DERIVED_TYPE(IDevicePreference, CPUDevicePreference, cpu);
AXON_SERIALIZE_DERIVED_TYPE(IDevicePreference, CudaDevicePreference, cuda);

void BindStruct(const aser::CStructBinder& binder, CPUDevicePreference& p)
{

}

void BindStruct(const aser::CStructBinder& binder, CudaDevicePreference& p)
{
    binder("deviceId", p.DeviceId);
}
