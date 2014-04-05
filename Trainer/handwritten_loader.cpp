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
	_labels = LoadLabels(labelFile);

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

MultiParams HandwrittenLoader::LoadImages(const std::string &file) const
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
	size_t numRows = read(fileStream, flipEndian);
	size_t numCols = read(fileStream, flipEndian);

	int imgSize = numRows * numCols;

	// Create the return value, initializing each image to the correct size
	MultiParams ret(numImages, Params(numCols, numRows, 1, Vector(imgSize)));

	unique_ptr<unsigned char[]> readBuf(new unsigned char[imgSize]);

	for (size_t i = 0; i < numImages; ++i)
	{
		fileStream.read((char *)readBuf.get(), imgSize);
		cvt_cpy(ret[i].Data.data(), readBuf.get(), imgSize);
	}

	return ret;
}
MultiParams HandwrittenLoader::LoadLabels(const std::string &file) const
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
	MultiParams ret(numLabels, Params(Vector::Zero(10)));

	for (size_t i = 0; i < numLabels; ++i)
	{
		char val = fileStream.get();

		// set the character using val as index to 1.0
		assert(val >= 0 && val <= 9);

		ret[i].Data[val] = 1.0;
	}

	return ret;
}
