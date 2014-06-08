#include "layer_base.h"

#include <stdexcept>

using namespace std;

LayerBase::LayerBase()
    : _net(nullptr)
{
    _devicePref = CPUDevicePreference::Instance;
}
LayerBase::LayerBase(std::string name)
    : _name(std::move(name)), _net(nullptr)
{
    _devicePref = CPUDevicePreference::Instance;
}

void LayerBase::InitializeFromConfig(const LayerConfig::Ptr &config)
{
}

LayerConfig::Ptr LayerBase::GetConfig() const
{
	auto ret = make_shared<LayerConfig>(_name);
	BuildConfig(*ret);
	return ret;
}

void LayerBase::BuildConfig(LayerConfig &config) const
{
	config.Name = _name;
}

Params* LayerBase::GetData(ParamMap &pMap, const string &name, bool enforce) const
{
	auto iter = pMap.find(name);

	if (iter != pMap.end())
		return &iter->second;

	if (enforce)
		throw runtime_error("The parameter '" + name + "' is not in the parameter map.");

	return nullptr;
}

const Params* LayerBase::GetData(const ParamMap &pMap, const string &name, bool enforce) const
{
	return GetData(const_cast<ParamMap&>(pMap), name, enforce);
}

void BindStruct(const aser::CStructBinder &binder, LayerBase &layer)
{
    binder("name", layer._name)
          ("device", layer._devicePref);

    if (binder.IsRead())
    {
        if (layer._name.empty())
           throw runtime_error("Cannot have a layer without a name.");

        layer.OnInitialized();
    }
}

void LayerBase::SetDevicePreference(IDevicePreference::Ptr pref)
{
	_devicePref = move(pref);

	OnInitialized();
}

void LayerBase::OnInitialized()
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
