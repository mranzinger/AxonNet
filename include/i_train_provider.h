#pragma once

#include "math_util.h"

class ITrainProvider
{
public:
	virtual ~ITrainProvider() { }

	virtual size_t Size() const = 0;
	virtual void Get(size_t idx, Vector &vals, Vector &labels) const = 0;

	virtual size_t TestSize() const = 0;
	virtual void GetTest(size_t idx, Vector &vals, Vector &labels) const = 0;
};