#pragma once

#include <memory>
#include <string>

#include <serialization/master.h>

#include "dll_include.h"
#include "math_util.h"
#include "params.h"

class NeuralNet;

struct NEURAL_NET_API ICost
{
public:
	typedef std::shared_ptr<ICost> Ptr;

	virtual ~ICost() { }

	virtual std::string GetType() const = 0;

	virtual Real Compute(const Params &pred, const Params &labels) = 0;
	virtual Params ComputeGrad(const Params &pred, const Params &labels) = 0;

	virtual void SetNet(NeuralNet *net) = 0;
};

// Default for most cost functions
inline void BindStruct(const axon::serialization::CStructBinder &, ICost &) { }

AXON_SERIALIZE_BASE_TYPE(ICost);
