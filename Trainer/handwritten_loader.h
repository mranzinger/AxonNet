#pragma once

#include <vector>

#include "i_train_provider.h"

class HandwrittenLoader
	: public ITrainProvider
{
private:
	std::vector<Params> _trainData;
	std::vector<Params> _labels;

	std::vector<Params> _testData;
	std::vector<Params> _testLabels;

public: 
	HandwrittenLoader(const std::string &dataFile, const std::string &labelFile,
		              const std::string &testDataFile, const std::string &testLabelFile);

	virtual size_t Size() const override {
		return _trainData.size();
	}
	virtual void Get(size_t idx, Params &vals, Params &labels) const override
	{
		vals = _trainData[idx];
		labels = _labels[idx];
	}

	virtual size_t TestSize() const override {
		return _testData.size();
	}
	virtual void GetTest(size_t idx, Params &vals, Params &labels) const override
	{
		vals = _testData[idx];
		labels = _testLabels[idx];
	}

private:
	MultiParams LoadImages(const std::string &file) const;
	MultiParams LoadLabels(const std::string &file) const;
};