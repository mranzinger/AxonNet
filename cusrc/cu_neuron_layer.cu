/*
 * File description: cu_neuron_layer.cu
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "cu_neuron_layer.cuh"

#include <stdexcept>

#include "cumat.cuh"
#include "cumath_unary_functions.cuh"
#include "cumath_binary_functions.cuh"
#include "cusetup_provider.cuh"

using namespace std;

template<typename UnaryFn>
void NCompute(const Params &input, Params &output, UnaryFn fn, CuContext handle)
{
    input.GetCudaMatrix(handle).UnaryExpr<false>(output.GetCudaMatrix(handle), fn);
}

void NCompute(const Params &input, Params &output, CuIdentity, CuContext handle)
{
    output = input;
}


void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
               CuIdentity, CuContext handle)
{
    inputErrs.GetCudaMatrix(handle).SetConstant(1.0f);
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
               CuLogistic, CuContext handle)
{
    lastOutput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuLogisticDerivativeCalc());
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
               CuRectifier, CuContext handle)
{
    lastInput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuRectifierDerivative());
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
               CuTanh, CuContext handle)
{
    lastOutput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuTanhDerivativeCalc());
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
                CuRamp, CuContext handle)
{
    lastInput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuRampDerivative());
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
            CuSoftplus, CuContext handle)
{
    lastInput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuSoftplusDerivativeRaw());
}

void NBackprop(const Params &lastInput, const Params &lastOutput, Params &inputErrs,
            CuHardTanh, CuContext handle)
{
    lastInput.GetCudaMatrix(handle).UnaryExpr<false>(inputErrs.GetCudaMatrix(handle),
            CuHardTanhDerivative());
}

template<typename CalcFn>
class CuNeuronLayer
    : public ICuNeuronLayer
{
private:
    CuContext _handle;

public:
    CuNeuronLayer(int deviceId)
    {
    	// TODO: Thread?
        _handle = CuSetupProvider::GetHandle(deviceId);
    }

    virtual Params Compute(const Params &input, bool isTraining)
    {
        CuMat *m = new CuMat(_handle, input.Rows, input.Cols);

        Params ret(input, m);

        NCompute(input, ret, CalcFn(), _handle);

        return ret;
    }
    virtual Params Backprop(const Params &input, const Params &lastOutput,
                            const Params &outputErrors)
    {
        CuMat *inputErrs = new CuMat(_handle, input.Rows, input.Cols);

        Params ret(input, inputErrs);

        NBackprop(input, lastOutput, ret, CalcFn(), _handle);

        inputErrs->CoeffMultiply(outputErrors.GetCudaMatrix(_handle));

        return ret;
    }
};

ICuNeuronLayer* CreateCuNeuronLayer(int deviceId, CuNeuronType type)
{
    switch (type)
    {
    case Cut_Linear:
        return new CuNeuronLayer<CuIdentity>(deviceId);
    case Cut_Logistic:
        return new CuNeuronLayer<CuLogistic>(deviceId);
    case Cut_Rectifier:
        return new CuNeuronLayer<CuRectifier>(deviceId);
    case Cut_Tanh:
        return new CuNeuronLayer<CuTanh>(deviceId);
    case Cut_Ramp:
        return new CuNeuronLayer<CuRamp>(deviceId);
    case Cut_SoftPlus:
        return new CuNeuronLayer<CuSoftplus>(deviceId);
    case Cut_HardTanh:
        return new CuNeuronLayer<CuHardTanh>(deviceId);
    default:
        throw runtime_error("Invalid neuron type.");
    }
}





















