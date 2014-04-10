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

using namespace std;

HandwrittenLoader::HandwrittenLoader(const string &root)
    : HandwrittenLoader(
       root + "train-images.idx3-ubyte",
	   root + "train-labels.idx1-ubyte",
	   root + "t10k-images.idx3-ubyte",
	   root + "t10k-labels.idx1-ubyte") 
{
    
}

HandwrittenLoader::HandwrittenLoader(const string &dataFile, const string &labelFile,
									 const string &testDataFile, const string &testLabelFile)
{
	_trainData = LoadImages(dataFile);
	_trainLabels = LoadLabels(labelFile);

	_testData = LoadImages(testDataFile);
	_testLabels = LoadLabels(testLabelFile);
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
HandwrittenLoader::MultiDataVec HandwrittenLoader::LoadLabels(const string &file)
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
	MultiDataVec ret(numLabels, DataVec(10, 0.0f));

	for (size_t i = 0; i < numLabels; ++i)
	{
		char val = fileStream.get();

		// set the character using val as index to 1.0
		assert(val >= 0 && val <= 9);

		ret[i][val] = 1.0;
	}

	return ret;
}

void HandwrittenLoader::Get(const vector<size_t> &idxs, Params &vals, Params &labels) const
{
	Get(idxs, vals, labels, _trainData, _trainLabels);
}

void HandwrittenLoader::GetTest(const vector<size_t> &idxs, Params &vals, Params &labels) const
{
	Get(idxs, vals, labels, _testData, _testLabels);
}

void HandwrittenLoader::Get(const std::vector<size_t>& idxs, Params& vals, Params& labels,
		const MultiDataVec& allImages,
		const MultiDataVec& allLabels) const
{
	vals.Width = _numRows;
	vals.Height = _numCols;
	vals.Depth = 1;

	// Each row is the ij-th pixel of each image
	vals.Data.resize(_imgSize, idxs.size());

	labels.Width = 10;
	labels.Height = 1;
	labels.Depth = 1;

	labels.Data.resize(10, idxs.size());

	// Since the data is stored column major, it means that each image
	// is stored in contiguous memory.
	Real *pVals = vals.Data.data();
	Real *pLabels = labels.Data.data();

	for (size_t idx : idxs)
	{
		const DataVec &vec = allImages[idx];
		const DataVec &lab = allLabels[idx];

		copy(vec.begin(), vec.end(), pVals);
		copy(lab.begin(), lab.end(), pLabels);

		pVals += _imgSize;
		pLabels += 10;
	}
}
