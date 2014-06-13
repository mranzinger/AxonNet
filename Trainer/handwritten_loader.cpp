#include "handwritten_loader.h"

#include <fstream>

#include <assert.h>

#if _WIN32
#include <filesystem>
#include <intrin.h>

namespace fs = tr2::sys;
#else
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
#endif

#include <Eigen/Dense>

using namespace std;
using namespace axon::serialization;

HandwrittenLoader::HandwrittenLoader(const string &root)
    : HandwrittenLoader(
       root + "train-images.idx3-ubyte",
	   root + "train-labels.idx1-ubyte",
	   root + "t10k-images.idx3-ubyte",
	   root + "t10k-labels.idx1-ubyte") 
{
    _rootDir = root;
}

HandwrittenLoader::HandwrittenLoader(const string &dataFile, const string &labelFile,
									 const string &testDataFile, const string &testLabelFile)
	: _dataFile(dataFile), _labelFile(labelFile), _testDataFile(testDataFile),
	  _testLabelFile(testLabelFile)
{
	Load();
}

template<typename T>
void read(istream &ip, T &val)
{
	ip.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template<typename T>
T read(istream &ip)
{
	T ret;
	read(ip, ret);
	return ret;
}

void read(istream &ip, int &val, bool flipEndian)
{
	read(ip, val);

	if (flipEndian)
	{
#if _WIN32
		val = _byteswap_ulong(val);
#else
		val = be32toh(val);
#endif
	}
}

int read(istream &ip, bool flipEndian)
{
	int ret;
	read(ip, ret, flipEndian);
	return ret;
}

void cvt_cpy(float *dstBuff, unsigned char *srcBuff, size_t buffSize)
{
	for (unsigned char *end = srcBuff + buffSize; srcBuff != end; ++dstBuff, ++srcBuff)
		*dstBuff = (((float) *srcBuff) / 256) - 0.5;
}

void HandwrittenLoader::Load()
{
	_trainData = LoadImages(_dataFile);
	_trainLabels = LoadLabels(_labelFile);

	_testData = LoadImages(_testDataFile);
	_testLabels = LoadLabels(_testLabelFile);

	Finalize();
}

HandwrittenLoader::MultiDataVec HandwrittenLoader::LoadImages(const std::string &file)
{
	static const int MAGIC_NUMBER = 0x00000803; // 2051

	if (!fs::exists(fs::path(file)))
		throw runtime_error("The specified file is not valid.");

	ifstream fileStream(file, ios_base::binary);

	if (fileStream.bad() || fileStream.eof())
		throw runtime_error("The specified file could not be opened, or was empty.");

	const auto magic = read<int>(fileStream);

	const bool flipEndian = magic != MAGIC_NUMBER;

	size_t numImages = read(fileStream, flipEndian);
	_numRows = read(fileStream, flipEndian);
	_numCols = read(fileStream, flipEndian);

	_imgSize = _numRows * _numCols;

	// Create the return value, initializing each image to the correct size
	MultiDataVec ret(numImages, DataVec(_imgSize));

	unique_ptr<unsigned char[]> readBuf(new unsigned char[_imgSize]);

	for (size_t i = 0; i < numImages; ++i)
	{
		fileStream.read((char *)readBuf.get(), _imgSize);
		cvt_cpy(ret[i].data(), readBuf.get(), _imgSize);
	}

	return ret;
}
HandwrittenLoader::LabelVec HandwrittenLoader::LoadLabels(const string &file)
{
	static const int MAGIC_NUMBER = 0x00000801; // 2049

	if (!fs::exists(fs::path(file)))
		throw runtime_error("The specified file is not valid.");

	ifstream fileStream(file, ios_base::binary);

	if (fileStream.bad() || fileStream.eof())
		throw runtime_error("The specified file could not be opened, or was empty.");

	const auto magic = read<int>(fileStream);

	const bool flipEndian = magic != MAGIC_NUMBER;

	size_t numLabels = read(fileStream, flipEndian);

	// Output is 10 values, [0, 1] for each character
	LabelVec ret(numLabels);

	for (size_t i = 0; i < numLabels; ++i)
	{
		char val = fileStream.get();

		// set the character using val as index to 1.0
		assert(val >= 0 && val <= 9);

		ret[i] = val;
	}

	return ret;
}

void HandwrittenLoader::GetTrain(ParamMap& inputMap, size_t a_batchSize)
{
    Get(GetTrainBatchIdxs(a_batchSize),
        inputMap,
        _trainData, _trainLabels, true);
}

void HandwrittenLoader::GetTest(ParamMap& inputMap, size_t a_offset,
        size_t a_batchSize)
{
    size_t numSamples = min(a_batchSize, TestSize() - a_offset);

    vector<size_t> idxs(numSamples);
    for (size_t i = 0; i < numSamples; ++i)
        idxs[i] = i + a_offset;

    Get(idxs, inputMap, _testData, _testLabels, false);
}

void HandwrittenLoader::Get(const vector<size_t>& idxs, ParamMap &inputMap,
		const MultiDataVec& allImages,
		const LabelVec& allLabels,
		bool deform) const
{
    Params vals(_numCols, _numRows, 1,
                new CMatrix(_imgSize, idxs.size()));
    Params labels(1, 10, 1, new CMatrix(CMatrix::Zero(10, idxs.size())));

	// Since the data is stored column major, it means that each image
	// is stored in contiguous memory.
	Real *pVals = vals.GetHostMatrix().data();
	Real *pLabels = labels.GetHostMatrix().data();

	std::random_device device;
	std::uniform_int_distribution<> dist(0, 3);

	RMatrix mat(28, 28);

	for (size_t idx : idxs)
	{
		const DataVec &vec = allImages[idx];
		size_t label = allLabels[idx];

		if (deform)
		{
			const RMap vMap(const_cast<Real*>(vec.data()), 28, 28);

			mat.setConstant(-0.5f);

			int wndX = dist(device);
			int wndY = dist(device);

			mat.block(wndY, wndX, 24, 24) = vMap.block(2, 2, 24, 24);

			copy(mat.data(), mat.data() + mat.size(), pVals);
		}
		else
		{
			copy(vec.begin(), vec.end(), pVals);
		}

		pLabels[label] = 1.0f;

		pVals += _imgSize;
		pLabels += 10;
	}

	TakeSet(inputMap, _inputName.empty() ? DEFAULT_INPUT_NAME : _inputName, vals);
	TakeSet(inputMap, _labelName.empty() ? DEFAULT_LABEL_NAME : _labelName, labels);
}

void WriteStruct(const CStructWriter &writer, const HandwrittenLoader &loader)
{
	if (!loader._rootDir.empty())
	{
		writer("rootDir", loader._rootDir);
	}
	else
	{
		writer("dataFile", loader._dataFile)
			  ("labelFile", loader._labelFile)
			  ("testDataFile", loader._testDataFile)
			  ("testLabelFile", loader._testLabelFile);
	}

	if (!loader._inputName.empty())
	    writer("inputName", loader._inputName);
	if (!loader._labelName.empty())
	    writer("labelName", loader._labelName);
}

void ReadStruct(const CStructReader &reader, HandwrittenLoader &loader)
{
	if (reader.GetData("rootDir"))
	{

	}
	else
	{

		reader("dataFile", loader._dataFile)
			  ("labelFile", loader._labelFile)
			  ("testDataFile", loader._testDataFile)
			  ("testLabelFile", loader._testLabelFile);
	}

	reader("inputName", loader._inputName)
	      ("labelName", loader._labelName);

	loader.Load();

	loader.Finalize();
}
