#pragma once

#include <Eigen/Dense>

#include <serialization/master.h>

#include "dll_include.h"
#include "math_util.h"

namespace axon {
	namespace serialization {

		NEURAL_NET_API void WriteStruct(const CStructWriter &writer, const Vector &vec);
		NEURAL_NET_API void WriteStruct(const CStructWriter &writer, const RMatrix &mat);

		NEURAL_NET_API void ReadStruct(const CStructReader &reader, Vector &vec);
		NEURAL_NET_API void ReadStruct(const CStructReader &reader, RMatrix &mat);

	}
}

