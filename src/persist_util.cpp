#include "persist_util.h"

#include <stdexcept>
#include <algorithm>

using namespace std;

namespace axon {
	namespace serialization {

		void WriteStruct(const CStructWriter &writer, const Vector &vec)
		{
			writer("size", vec.size());

			CPrimArrayData<Real>::Ptr vals(new CPrimArrayData<Real>(writer.GetContext()));

			vals->Import(vec.data(), vec.data() + vec.size());

			writer.Append("data", move(vals));
		}
		void WriteStruct(const CStructWriter &writer, const Matrix &mat)
		{
			writer("rows", mat.outerSize());
			writer("cols", mat.innerSize());

			CPrimArrayData<Real>::Ptr vals(new CPrimArrayData<Real>(writer.GetContext()));

			vals->Import(mat.data(), mat.data() + mat.size());

			writer.Append("data", move(vals));
		}

		void ReadStruct(const CStructReader &reader, Vector &vec)
		{
			size_t sz;
			reader("size", sz);

			vec.resize(sz);

			auto data = dynamic_cast<const CPrimArrayData<Real> *>(reader.GetData("data"));

			if (!data)
				throw runtime_error("Couldn't deserialize the data vector because it wasn't present in the storage.");

			copy(data->begin(), data->end(), vec.data());
		}
		void ReadStruct(const CStructReader &reader, Matrix &mat)
		{
			size_t rows, cols;
			reader("rows", rows)("cols", cols);

			mat.resize(rows, cols);

			auto data = dynamic_cast<const CPrimArrayData<Real> *>(reader.GetData("data"));

			if (!data)
				throw runtime_error("Couldn't deserialize the data vector because it wasn't present in the storage.");

			copy(data->begin(), data->end(), mat.data());
		}
	}
}