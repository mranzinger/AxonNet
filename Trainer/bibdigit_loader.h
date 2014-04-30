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

	MultiDataVec _trainData;
	MultiDataVec _trainLabels;

	MultiDataVec _testData;
	MultiDataVec _testLabels;

	std::string _rootDir;

public:
	BibDigitLoader() = default;
	BibDigitLoader(const std::string &rootDir);

	virtual size_t Size() const override { return _trainData.size(); }
	virtual void Get(const std::vector<size_t> &idxs, Params &vals, Params &labels) const override;

	virtual size_t TestSize() const override { return _testData.size(); }
	virtual void GetTest(const std::vector<size_t> &idxs, Params &vals, Params &labels) const override;

	friend void WriteStruct(const axon::serialization::CStructWriter &writer, const BibDigitLoader &loader);
	friend void ReadStruct(const axon::serialization::CStructReader &reader, BibDigitLoader &loader);

private:
	void LoadDataset(const std::string &file, MultiDataVec &data, MultiDataVec &labels) const;
};



#endif /* BIBDIGIT_LOADER_H_ */
