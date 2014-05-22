/*
 * File description: bibdigit_loader.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#ifndef BIBDIGIT_LOADER_H_
#define BIBDIGIT_LOADER_H_

#include <vector>

#include "i_train_provider.h"

class BibDigitLoader
	: public ITrainProvider
{
private:
	typedef std::vector<Real> DataVec;
	typedef std::vector<DataVec> MultiDataVec;
	typedef std::vector<size_t> LabelVec;

	MultiDataVec _trainData;
	LabelVec _trainLabels;

	MultiDataVec _testData;
	LabelVec _testLabels;

	std::string _rootDir;

public:
	BibDigitLoader() = default;
	BibDigitLoader(const std::string &rootDir);

	virtual size_t TrainSize() const override { return _trainData.size(); }
	virtual size_t TestSize() const override { return _testData.size(); }

	virtual void GetTrain(ParamMap &inputMap, size_t a_batchSize) override;
    virtual void GetTest(ParamMap &inputMap, size_t a_offset, size_t a_batchSize) override;

	friend void WriteStruct(const aser::CStructWriter &writer, const BibDigitLoader &loader);
	friend void ReadStruct(const aser::CStructReader &reader, BibDigitLoader &loader);

private:
	void LoadDataset(const std::string &file, MultiDataVec &data, LabelVec &labels) const;
};



#endif /* BIBDIGIT_LOADER_H_ */
