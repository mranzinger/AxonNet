/*
 * File description: device_preference.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <serialization/master.h>

namespace aser = axon::serialization;

enum class DevicePreference
{
    CPU,
    Cuda,
    OpenCL
};

class IDevicePreference
{
public:
    typedef std::shared_ptr<IDevicePreference> Ptr;

    virtual ~IDevicePreference() { }

    virtual DevicePreference Type() const = 0;
};

class CPUDevicePreference
    : public IDevicePreference
{
public:
    typedef std::shared_ptr<CPUDevicePreference> Ptr;

    static Ptr Instance;

    virtual DevicePreference Type() const
    {
        return DevicePreference::CPU;
    }
};

class CudaDevicePreference
    : public IDevicePreference
{
public:
    typedef std::shared_ptr<CudaDevicePreference> Ptr;

    int DeviceId;

    CudaDevicePreference()
        : DeviceId(0) { }

    virtual DevicePreference Type() const
    {
        return DevicePreference::Cuda;
    }

};

void BindStruct(const aser::CStructBinder &binder, CPUDevicePreference &p);
void BindStruct(const aser::CStructBinder &binder, CudaDevicePreference &p);

AXON_SERIALIZE_BASE_TYPE(IDevicePreference);
