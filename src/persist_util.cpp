#include "persist_util.h"

#include <stdexcept>
#include <algorithm>

using namespace std;

namespace axon {
	namespace serialization {
	    namespace {
	        template<typename MatType>
	        void WriteMat(const CStructWriter &writer, const MatType &mat)
	        {
	            writer("rows", mat.rows())
                      ("cols", mat.cols());

                CPrimArrayData<Real>::Ptr vals(new CPrimArrayData<Real>(writer.GetContext()));

                vals->Import(mat.data(), mat.data() + mat.size());

                writer.Append("data", move(vals));
	        }
	        template<typename MatType>
	        void ReadMat(const CStructReader &reader, MatType &mat)
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


		void WriteStruct(const CStructWriter &writer, const Vector &vec)
		{
			writer("size", vec.size());

			CPrimArrayData<Real>::Ptr vals(new CPrimArrayData<Real>(writer.GetContext()));

			vals->Import(vec.data(), vec.data() + vec.size());

			writer.Append("data", move(vals));
		}
		void WriteStruct(const CStructWriter &writer, const RMatrix &mat)
		{
		    WriteMat(writer, mat);
		}
		void WriteStruct(const CStructWriter& writer, const CMatrix& mat)
		{
		    WriteMat(writer, mat);
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
		void ReadStruct(const CStructReader &reader, RMatrix &mat)
		{
		    ReadMat(reader, mat);
		}
		void ReadStruct(const CStructReader& reader, CMatrix& mat)
		{
		    ReadMat(reader, mat);
		}
	}
}




