#pragma once

#include <serialization/master.h>

#include "params.h"

class ITrainProvider
{
public:
	virtual ~ITrainProvider() { }

	virtual size_t Size() const = 0;
	virtual void Get(const std::vector<size_t> &idxs, Params &vals, Params &labels) const = 0;

	virtual size_t TestSize() const = 0;
	virtual void GetTest(const std::vector<size_t> &idxs, Params &vals, Params &labels) const = 0;
};

AXON_SERIALIZE_BASE_TYPE(ITrainProvider);
