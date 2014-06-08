#pragma once

#include <memory>
#include <string>
#include <map>

#include <serialization/master.h>

#include "dll_include.h"
#include "math_util.h"
#include "params.h"

#include "device_preference.h"

class NeuralNet;

namespace aser = axon::serialization;

class CostMap
    : public std::map<std::string, Real>
{
public:
    static const std::string PRIMARY_NAME;

    CostMap() = default;
    CostMap(std::initializer_list<value_type> initList);

    CostMap &operator+=(const CostMap &mp);
    CostMap &operator*=(Real val);
    CostMap &operator/=(Real val);

    friend void BindStruct(const aser::CStructBinder &binder, CostMap &mp);
};

class NEURAL_NET_API ICost
{
public:
	typedef std::shared_ptr<ICost> Ptr;

	virtual ~ICost() { }

	virtual std::string GetType() const = 0;

	virtual CostMap Compute(const ParamMap &inputs) = 0;
	virtual void ComputeGrad(const ParamMap &inputs, ParamMap &inputErrors) = 0;

	virtual bool IsBetter(const CostMap &a, const CostMap &b) const = 0;

	virtual void SetNet(NeuralNet *net) = 0;

	virtual void SetDevicePreference(IDevicePreference::Ptr pref) = 0;
};

// Default for most cost functions
inline void BindStruct(const aser::CStructBinder &, ICost &) { }

AXON_SERIALIZE_BASE_TYPE(ICost);
