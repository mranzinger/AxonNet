/*
 * convo_layer_test.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#include <gtest/gtest.h>

#include "convo_layer.h"

#include "test_helper.h"

Params Compute(const RMatrix &kernel, const Vector &bias,
			   const Params &input,
			   size_t windowSizeX, size_t windowSizeY,
			   size_t strideX, size_t strideY,
			   size_t padWidth, size_t padHeight,
			   bool isTraining = false)
{
	return ConvoLayer("",
					  kernel, bias,
					  windowSizeX, windowSizeY,
					  strideX, strideY,
					  padWidth, padHeight)
			.Compute(0, input, isTraining);
}

TEST(ConvoLayerTest, SimpleConvo)
{
	RMatrix kernel(1, 9); // 1 Filter, 9 inputs
	kernel << -1, 0, 1,
			  -2, 0, 2,
			  -1, 0, 1;
	Vector bias(1);
	bias << 1;

	CMatrix input(9, 2); // 9 pixels per image, 2 images
	input << 1, 3,
			 2, 2,
			 3, 1,
			 1, 3,
			 2, 2,
			 3, 1,
			 1, 3,
			 2, 2,
			 3, 1;

	CMatrix correctOutput(1, 2);
	correctOutput << 9, -7;

	Params comp = Compute(kernel, bias, Params(3, 3, 1, input),
						  3, 3,
						  1, 1,
						  0, 0);

	AssertMatrixEquivalence(correctOutput, comp.Data);
}


