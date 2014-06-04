/*
 * File description: cumath_traits.cuh
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include "cudev_helper.cuh"
#include "math_defines.h"

template<typename T>
class has_coord_functor
{
private:
    template<typename U, U>
    class check { };

    template<typename C>
    static char f(check<Real (C::*)(Real,uint32_t,uint32_t) const, &C::operator()>*);
    template<typename C>
    static char f(check<Real (C::*)(Real,uint32_t,uint32_t), &C::operator()>*);
    template<typename C>
    static char f(check<Real (C::*)(Real,Real,uint32_t,uint32_t) const, &C::operator()>*);
    template<typename C>
    static char f(check<Real (C::*)(Real,Real,uint32_t,uint32_t), &C::operator()>*);
    template<typename C>
    static char f(check<Real (C::*)(Real,Real,Real,uint32_t,uint32_t) const, &C::operator()>*);
    template<typename C>
    static char f(check<Real (C::*)(Real,Real,Real,uint32_t,uint32_t), &C::operator()>*);

    template<typename C>
    static long f(...);

public:
    static const bool value = sizeof(f<T>(0)) == sizeof(char);
};

template<bool Req>
struct CuFunctorTraits_t
{
    // Informs the CUDA kernel on whether or not the function
    // requires the coordinates to function properly.
    // If it does, the expected function signature is:
    // (Real val..., uint32_t row, uint32_t col)
    // Otherwise:
    // (Real val...)
    static const bool RequiresCoordinates = Req;
};

template<typename T>
struct CuFunctorTraits
    : CuFunctorTraits_t<has_coord_functor<T>::value>
{
};
