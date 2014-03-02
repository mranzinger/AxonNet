#pragma once

#include <vector>

#include "i_train_provider.h"

class HandwrittenLoader
	: public ITrainProvider
{
private:
	std::vector<Vector> _trainData;
	std::vector<Vector> _labels;

	std::vector<Vector> _testData;
	std::vector<Vector> _testLabels;

public: 
	HandwrittenLoader(const std::string &dataFile, const std::string &labelFile,
		              const std::string &testDataFile, const std::string &testLabelFile);

	virtual size_t Size() const override {
		return _trainData.size();
	}
	virtual void Get(size_t idx, Vector &vals, Vector &labels) const override
	{
		vals = _trainData[idx];
		labels = _labels[idx];
	}

	virtual size_t TestSize() const override {
		return _testData.size();
	}
	virtual void GetTest(size_t idx, Vector &vals, Vector &labels) const override
	{
		vals = _testData[idx];
		labels = _testLabels[idx];
	}

private:
	std::vector<Vector> LoadImages(const std::string &file) const;
	std::vector<Vector> LoadLabels(const std::string &file) const;
};