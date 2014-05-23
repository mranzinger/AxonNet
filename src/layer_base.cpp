#include "layer_base.h"

#include <stdexcept>

using namespace std;

LayerBase::LayerBase(std::string name)
    : _name(std::move(name)), _net(nullptr)
{
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
    binder("name", layer._name);

    if (binder.IsRead() && layer._name.empty())
        throw runtime_error("Cannot have a layer without a name.");
}
