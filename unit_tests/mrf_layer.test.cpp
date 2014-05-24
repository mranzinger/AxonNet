/*
 * mrf_layer.test.cpp
 *
 *  Created on: May 24, 2014
 *      Author: mike
 */

#include <vector>

#include <gtest/gtest.h>

#include "mrf_layer.h"

#include "test_helper.h"

namespace {

inline Params Compute(const Params &input,
			   size_t width, size_t height,
			   bool isTraining = false)
{
	ParamMap mp{ { "test-input", input } };

	MRFLayer("test-output", "test-input", width, height).Compute(mp, isTraining);

	return mp["test-output"];
}

inline Params Backprop(const Params &lastInput, const Params &outputErrors,
			    size_t width, size_t height)
{
	ParamMap inputs{ { "test-input", lastInput } };
	ParamMap errs{ { "test-output", outputErrors } };

	MRFLayer("test-output", "test-input", width, height).Backprop(inputs, errs);

	return errs["test-input"];
}

}

TEST(MRFLayerTest, ComputeSingularity)
{
	CMatrix input(9, 1);
	input << 1,  1, -2,
			 5, -3,  1,
			 4,  4,  2;

	CMatrix correctOutput(4, 1);
	correctOutput << 5, -3,
			         4,  4;

	Params comp = Compute(Params(3, 3, 1, input), 2, 2);

	AssertMatrixEquivalence(correctOutput, comp.Data);
}

