#include "handwritten_loader.h"

#include <fstream>
#include <filesystem>
#include <intrin.h>
#include <assert.h>

using namespace std;

namespace fs = tr2::sys;

HandwrittenLoader::HandwrittenLoader(const std::string &dataFile, const std::string &labelFile,
									 const std::string &testDataFile, const std::string &testLabelFile)
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
		val = _byteswap_ulong(val);
}

int read(istream &ip, bool flipEndian)
{
	int ret;
	read(ip, ret, flipEndian);
	return ret;
}

void cvt_cpy(float *dstBuff, char *srcBuff, size_t buffSize)
{
	for (char *end = srcBuff + buffSize; srcBuff != end; ++dstBuff, ++srcBuff)
		*dstBuff = ((float) *srcBuff) / 255.0;
}

vector<Vector> HandwrittenLoader::LoadImages(const std::string &file) const
{
	static const int MAGIC_NUMBER = 0x00000803; // 2051

	if (!fs::exists(fs::path(file)))
		throw exception("The specified file is not valid.");

	ifstream fileStream(file, ios_base::binary);

	if (fileStream.bad() || fileStream.eof())
		throw exception("The specified file could not be opened, or was empty.");

	const auto magic = read<int>(fileStream);

	const bool flipEndian = magic != MAGIC_NUMBER;

	int numImages = read(fileStream, flipEndian);
	int numRows = read(fileStream, flipEndian);
	int numCols = read(fileStream, flipEndian);

	int imgSize = numRows * numCols;

	// Create the return value, initializing each image to the correct size
	vector<Vector> ret(numImages, Vector(imgSize));

	auto readBuf = make_unique<char[]>(imgSize);

	for (size_t i = 0; i < numImages; ++i)
	{
		fileStream.read(readBuf.get(), imgSize);
		cvt_cpy(ret[i].data(), readBuf.get(), imgSize);
	}

	return ret;
}
std::vector<Vector> HandwrittenLoader::LoadLabels(const std::string &file) const
{
	static const int MAGIC_NUMBER = 0x00000801; // 2049

	if (!fs::exists(fs::path(file)))
		throw exception("The specified file is not valid.");

	ifstream fileStream(file, ios_base::binary);

	if (fileStream.bad() || fileStream.eof())
		throw exception("The specified file could not be opened, or was empty.");

	const auto magic = read<int>(fileStream);

	const bool flipEndian = magic != MAGIC_NUMBER;

	int numLabels = read(fileStream, flipEndian);

	// Output is 10 values, [0, 1] for each character
	vector<Vector> ret(numLabels, Vector::Zero(10));

	for (size_t i = 0; i < numLabels; ++i)
	{
		char val = fileStream.get();

		// set the character using val as index to 1.0
		assert(val >= 0 && val <= 9);

		ret[i][val] = 1.0;
	}

	return ret;
}
