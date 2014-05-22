/*
 * File description: train_provider_base.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "train_provider_base.h"

using namespace std;

TrainProviderBase::TrainProviderBase()
{
    random_device rnddev;
    _trainRand.seed(rnddev());
}

void TrainProviderBase::Finalize()
{
    _trainDist = uniform_int_distribution<size_t>(0, TrainSize() - 1);
}

vector<size_t> TrainProviderBase::GetTrainBatchIdxs(size_t batchSize)
{
    vector<size_t> vec;
    GetTrainBatchIdxs(vec, batchSize);
    return move(vec);
}

void TrainProviderBase::GetTrainBatchIdxs(vector<size_t> &vec, size_t batchSize)
{
    vec.resize(batchSize);

    for (size_t &val : vec)
        val = GetNextTrain();
}
