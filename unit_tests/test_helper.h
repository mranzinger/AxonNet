/*
 * test_helper.h
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#pragma once

#include <iostream>
#include <string>
#include <sstream>

#include "math_defines.h"

extern const Real DEFAULT_PRECISION;

void AssertVectorEquivalence(const Vector &a, const Vector &b, Real precision = DEFAULT_PRECISION);
void AssertMatrixEquivalence(const CMatrix &a, const CMatrix &b, Real precision = DEFAULT_PRECISION);
void AssertMatrixEquivalence(const RMatrix &a, const RMatrix &b, Real precision = DEFAULT_PRECISION);


template<typename T>
std::string eig_to_str(const T &a)
{
    std::ostringstream ss;

    ss << a;

    return ss.str();
}

std::string MatToStr(const Vector &a);
std::string MatToStr(const CMatrix &a);
std::string MatToStr(const RMatrix &a);
