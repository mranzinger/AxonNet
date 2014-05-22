/*
 * File description: train_provider_base.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <random>
#include <vector>

#include "i_train_provider.h"

class NEURAL_NET_API TrainProviderBase
    : public virtual ITrainProvider
{
scope_protected:
    std::mt19937_64 _trainRand;
    std::uniform_int_distribution<size_t> _trainDist;

scope_public:
    TrainProviderBase();

    virtual void Finalize() override;

scope_protected:
    inline size_t GetNextTrain()
    {
        return _trainDist(_trainRand);
    }
    std::vector<size_t> GetTrainBatchIdxs(size_t batchSize);
    void GetTrainBatchIdxs(std::vector<size_t> &vec, size_t batchSize);
};
