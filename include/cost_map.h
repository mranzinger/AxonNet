/*
 * cost_map.h
 *
 *  Created on: Jun 8, 2014
 *      Author: mike
 */


#pragma once

#include <map>

#include "math_defines.h"

#ifndef _CUDA_COMPILE_
#include <serialization/master.h>

namespace aser = axon::serialization;
#endif

class CostMap
    : public std::map<std::string, Real>
{
public:
    static const std::string PRIMARY_NAME;

#ifndef _CUDA_COMPILE_
    CostMap() = default;
    CostMap(std::initializer_list<value_type> initList);
#endif

    CostMap &operator+=(const CostMap &mp);
    CostMap &operator*=(Real val);
    CostMap &operator/=(Real val);

#ifndef _CUDA_COMPILE_
    friend void BindStruct(const aser::CStructBinder &binder, CostMap &mp);
#endif
};


