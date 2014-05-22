#pragma once

#include <serialization/master.h>

#include "dll_include.h"
#include "params.h"

namespace aser = axon::serialization;

class NEURAL_NET_API ITrainProvider
{
public:
	virtual ~ITrainProvider() { }

	static const std::string DEFAULT_INPUT_NAME;
	static const std::string DEFAULT_LABEL_NAME;

	virtual size_t TrainSize() const = 0;
	virtual size_t TestSize() const = 0;

	virtual void GetTrain(ParamMap &inputMap, size_t a_batchSize) = 0;
	virtual void GetTest(ParamMap &inputMap, size_t a_offset, size_t a_batchSize) = 0;

	virtual void Finalize() = 0;
};

AXON_SERIALIZE_BASE_TYPE(ITrainProvider);
