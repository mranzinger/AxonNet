/*
 * File description: cumat_temp_agg.cuh
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "cumat.cuh"

#include <stdint.h>

#include "cudev_helper.cuh"

namespace {

template<uint32_t RowInc, uint32_t ColInc>
class Incrementer
{
    __device__ __host__ void operator()(uint32_t &row, uint32_t &col) const
    {
        row += RowInc;
        col += ColInc;
    }
};

template<typename Inc, typename Aggregator, typename ElemFn>
CuMat CuMatAggregate(const CuMat &mat,
                     CuMatInfo matInfo, Inc inc, Aggregator agg, ElemFn fn)
{

}

}

template<typename ElemFn>
CuMat CuRowwiseOperator::Sum(ElemFn fn) const
{
    return Agg(CuPlus(), fn);
}

template<typename ElemFn>
CuMat CuColwiseOperator::Sum(ElemFn fn) const
{
    return Agg(CuPlus(), fn);
}

template<typename Aggregator, typename ElemFn>
CuMat CuRowwiseOperator::Agg(Aggregator agg, ElemFn fn) const
{
    return CuMatAggregate(Mat, Mat.ToInfo(),
                          Incrementer<0, 1>(),
                          agg, fn);
}

template<typename Aggregator, typename ElemFn>
CuMat CuColwiseOperator::Agg(Aggregator agg, ElemFn fn) const
{
    return CuMatAggregate(Mat, Mat.ToInfo(),
                          Incrementer<1, 0>(),
                          agg, fn);
}
