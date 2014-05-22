#pragma once

#include <vector>

#include "2d_train_provider_base.h"

class HandwrittenLoader
	: public C2dTrainProviderBase
{
private:
	typedef std::vector<Real> DataVec;
	typedef std::vector<DataVec> MultiDataVec;
	typedef std::vector<size_t> LabelVec;

	MultiDataVec _trainData;
	LabelVec _trainLabels;

	MultiDataVec _testData;
	LabelVec _testLabels;

	size_t _numRows;
	size_t _numCols;
	size_t _imgSize;

	std::string _rootDir,
				_dataFile, _labelFile,
				_testDataFile, _testLabelFile;
	std::string _inputName,
	            _labelName;

public: 
	HandwrittenLoader() = default;
	HandwrittenLoader(const std::string &rootDir);
    HandwrittenLoader(const std::string &dataFile, const std::string &labelFile,
		              const std::string &testDataFile, const std::string &testLabelFile);

	virtual size_t TrainSize() const override {
		return _trainData.size();
	}
	virtual size_t TestSize() const override {
		return _testData.size();
	}

	virtual void GetTrain(ParamMap &inputMap, size_t a_batchSize) override;
    virtual void GetTest(ParamMap &inputMap, size_t a_offset, size_t a_batchSize) override;

	friend void WriteStruct(const aser::CStructWriter &writer, const HandwrittenLoader &loader);
	friend void ReadStruct(const aser::CStructReader &reader, HandwrittenLoader &loader);

private:
	void Load();

	MultiDataVec LoadImages(const std::string &file);
	MultiDataVec LoadLabels(const std::string &file);

	void Get(const std::vector<size_t> &idxs, ParamMap &inputMap,
			 const MultiDataVec &allImages, const LabelVec &allLabels, bool deform) const;
};
