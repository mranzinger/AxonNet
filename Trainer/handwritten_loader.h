#pragma once

#include <vector>

#include "i_train_provider.h"

class HandwrittenLoader
	: public ITrainProvider
{
private:
	typedef std::vector<Real> DataVec;
	typedef std::vector<DataVec> MultiDataVec;

	MultiDataVec _trainData;
	MultiDataVec _trainLabels;

	MultiDataVec _testData;
	MultiDataVec _testLabels;

	size_t _numRows;
	size_t _numCols;
	size_t _imgSize;

public: 
	HandwrittenLoader(const std::string &rootDir);
    HandwrittenLoader(const std::string &dataFile, const std::string &labelFile,
		              const std::string &testDataFile, const std::string &testLabelFile);

	virtual size_t Size() const override {
		return _trainData.size();
	}
	virtual void Get(const std::vector<size_t> &idxs, Params &vals, Params &labels) const override;

	virtual size_t TestSize() const override {
		return _testData.size();
	}
	virtual void GetTest(const std::vector<size_t> &idxs, Params &vals, Params &labels) const override;

private:
	MultiDataVec LoadImages(const std::string &file);
	MultiDataVec LoadLabels(const std::string &file);

	void Get(const std::vector<size_t> &idxs, Params &vals, Params &labels,
			 const MultiDataVec &allImages, const MultiDataVec &allLabels) const;
};
