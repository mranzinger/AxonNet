/*
 * File description: bibdigit_loader.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "bibdigit_loader.h"

BibDigitLoader::BibDigitLoader(const std::string& rootDir)
{
}

void BibDigitLoader::GetTrain(ParamMap& inputMap, size_t a_batchSize)
{
}

void BibDigitLoader::GetTest(ParamMap& inputMap, size_t a_offset,
        size_t a_batchSize)
{
}

void BibDigitLoader::LoadDataset(const std::string& file, MultiDataVec& data,
        LabelVec& labels) const
{
}



void ReadStruct(const aser::CStructReader &reader, BibDigitLoader &loader)
{

}

void WriteStruct(const aser::CStructWriter &writer, const BibDigitLoader &loader)
{

}

