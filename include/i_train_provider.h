#pragma once

#include "params.h"

class ITrainProvider
{
public:
	virtual ~ITrainProvider() { }

	virtual size_t Size() const = 0;
	virtual void Get(size_t idx, Params &vals, Params &labels) const = 0;

	virtual size_t TestSize() const = 0;
	virtual void GetTest(size_t idx, Params &vals, Params &labels) const = 0;
};