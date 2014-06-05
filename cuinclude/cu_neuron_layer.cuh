/*
 * File description: cu_neuron_layer.cuh
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "params.h"

enum CuNeuronType
{
    Cut_Linear,
    Cut_Logistic,
    Cut_Rectifier,
    Cut_Tanh,
    Cut_Ramp,
    Cut_SoftPlus,
    Cut_HardTanh
};

class ICuNeuronLayer
{
scope_public:
    virtual ~ICuNeuronLayer() { }

    virtual Params Compute(const Params &input, bool isTraining) = 0;
    virtual Params Backprop(const Params &input, const Params &lastOutput,
                            const Params &outputErrors) = 0;
};

ICuNeuronLayer *CreateCuNeuronLayer(int deviceId, CuNeuronType type);

