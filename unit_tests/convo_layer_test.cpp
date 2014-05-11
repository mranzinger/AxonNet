/*
 * convo_layer_test.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: mike
 */

#include <vector>

#include <gtest/gtest.h>

#include "convo_layer.h"

#include "test_helper.h"

using namespace std;

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

Params Backprop(const RMatrix &kernel, const Vector &bias,
			    const Params &lastInput, const Params &outputErrors,
			    size_t windowSizeX, size_t windowSizeY,
			    size_t strideX, size_t strideY,
			    size_t padWidth, size_t padHeight,
			    RMatrix &weightsGrad, Vector &biasGrad,
			    const Params *pLastOutput = nullptr)
{
	ConvoLayer layer("", kernel, bias,
					 windowSizeX, windowSizeY,
					 strideX, strideY,
					 padWidth, padHeight);

	const Params &lastOutput = pLastOutput ?
								  *pLastOutput
							    : layer.Compute(0, lastInput, true);

	Params ret = layer.Backprop(0, lastInput, lastOutput, outputErrors);

	weightsGrad = layer._linearLayer.GetParams(0).WeightsGrad;
	biasGrad = layer._linearLayer.GetParams(0).BiasGrad;

	return move(ret);
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

TEST(ConvoLayerTest, ConvoMultiFilter)
{
	RMatrix kernel(2, 4); // 2 Filters, 4 inputs
	kernel << 1, 1, 1, 1,
			  1, 2, 3, 4;
	Vector bias(2);
	bias << -4, 1;

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

	CMatrix correctOutput(8, 2);
	correctOutput <<  2,  6,     17, 25,
			          6,  2,     27, 15,

					  2,  6,     17, 25,
					  6,  2,     27, 15;

	Params comp = Compute(kernel, bias, Params(3, 3, 1, input),
						  2, 2,
						  1, 1,
						  0, 0);

	AssertMatrixEquivalence(correctOutput, comp.Data);
}

TEST(ConvoLayerTest, ConvoPaddedSimple)
{
	RMatrix kernel(1, 9);
	kernel << 1, 1, 1,
			  1, 1, 1,
			  1, 1, 1;
	Vector bias(1);
	bias << -1;

	CMatrix input(1, 2);
	input << 1, 2;

	CMatrix correctOutput(1, 2);
	correctOutput << 0, 1;

	Params comp = Compute(kernel, bias, Params(1, 1, 1, input),
						  3, 3,
						  1, 1,
						  1, 1);

	AssertMatrixEquivalence(correctOutput, comp.Data);
}

TEST(ConvoLayerTest, ConvoPaddedHarder)
{
	RMatrix kernel(1, 9);
	kernel << 1, 2, 3,
			  4, 5, 6,
			  7, 8, 9;
	Vector bias(1);
	bias << 0;

	CMatrix input(9, 1);
	input << 1, 2, 3,
			 4, 5, 6,
			 7, 8, 9;

	CMatrix correctOutput(4, 1);
	correctOutput << (5*1 + 6*2 + 8*4 + 9*5), (4*2 + 5*3 + 7*5 + 8*6),
			         (2*4 + 3*5 + 5*7 + 6*8), (1*5 + 2*6 + 4*8 + 5*9);

	Params comp = Compute(kernel, bias, Params(3, 3, 1, input),
						  3, 3,
						  2, 2,
						  1, 1);

	AssertMatrixEquivalence(correctOutput, comp.Data);
}

TEST(ConvoLayerTest, SimpleBackprop)
{
	RMatrix kernel(1, 9);
	kernel << 1, 1, 1,
			  1, 1, 1,
			  1, 1, 1;
	Vector bias(1);
	bias << 0;

	CMatrix input(9, 1);
	input << 1, 2, 3,
			 4, 5, 6,
			 7, 8, 9;

	CMatrix outputErrors(1, 1);
	outputErrors << -5;

	RMatrix weightsGrad;
	Vector biasGrad;

	Params inputErrors = Backprop(kernel, bias,
								  Params(3, 3, 1, input),
								  Params(1, 1, 1, outputErrors),
								  3, 3,
								  1, 1,
								  0, 0,
								  weightsGrad, biasGrad);

	// Since there will only be one application of the kernel over
	// the entire span of the input, the input error should be the same
	// as what the linear layer computes
	LinearLayer validator("", kernel, bias);

	Params linInputErrors = validator.UTBackprop(0,
											   input,
											   outputErrors);

	AssertMatrixEquivalence(inputErrors.Data, linInputErrors.Data);

	AssertMatrixEquivalence(validator.GetParams(0).WeightsGrad,
							weightsGrad);

	AssertVectorEquivalence(validator.GetParams(0).BiasGrad,
							biasGrad);
}

TEST(ConvoLayerTest, HarderBackprop)
{
	RMatrix kernel(1, 9);
	kernel << 1, 1, 1,
			  1, 1, 1,
			  1, 1, 1;
	Vector bias(1);
	bias << 0;

	CMatrix input(25, 1);
	input << 1, 1, 1, 1, 1,
			 1, 2, 2, 2, 1,
			 1, 2, 3, 2, 1,
			 1, 2, 2, 2, 1,
			 1, 1, 1, 1, 1;

	CMatrix outputErrors(9, 1);
	outputErrors << 1, 1, 1,
					1, -3, 1,
					1, 1, 1;

	RMatrix weightsGrad;
	Vector biasGrad;

	Params inputErrors = Backprop(kernel, bias,
								  Params(5, 5, 1, input),
								  Params(3, 3, 1, outputErrors),
								  3, 3,
								  1, 1,
								  0, 0,
								  weightsGrad, biasGrad);

	// Bit harder here to use the linear layer, but it is still less error prone
	// than manual computation
	LinearLayer validator("", kernel, bias);

	CMatrix cInputErrors = CMatrix::Zero(25, 1);
	CMap cMapInputErrors(cInputErrors.data(), 5, 5);
	CMap mapInput(input.data(), 5, 5);

	for (int y = 0; y < 3; ++y)
	{
		for (int x = 0; x < 3; ++x)
		{
			RMatrix bInWindow = mapInput.block(y, x, 3, 3);

			CMatrix inWindow = RMap(bInWindow.data(), 9, 1);

			CMatrix opErrWindow = outputErrors.block(y * 3 + x, 0, 1, 1);

			Params linInputErrors = validator.UTBackprop(0,
													   inWindow,
													   opErrWindow);

			CMatrix linErrWindow = RMap(linInputErrors.Data.data(), 3, 3);

			cMapInputErrors.block(y, x, 3, 3) += linErrWindow;
		}
	}

	AssertMatrixEquivalence(inputErrors.Data, cInputErrors);
}

TEST(ConvoLayerTest, HarderBackpropStride)
{
	RMatrix kernel(1, 9);
	kernel << 1, 1, 1,
			  1, 1, 1,
			  1, 1, 1;
	Vector bias(1);
	bias << 0;

	CMatrix input(25, 1);
	input << 1, 1, 1, 1, 1,
			 1, 2, 2, 2, 1,
			 1, 2, 3, 2, 1,
			 1, 2, 2, 2, 1,
			 1, 1, 1, 1, 1;

	CMatrix outputErrors(9, 1);
	outputErrors << 1, 1, 1,
					1, -3, 1,
					1, 1, 1;

	RMatrix weightsGrad;
	Vector biasGrad;

	Params inputErrors = Backprop(kernel, bias,
								  Params(5, 5, 1, input),
								  Params(3, 3, 1, outputErrors),
								  3, 3,
								  2, 2,
								  0, 0,
								  weightsGrad, biasGrad);

	// Bit harder here to use the linear layer, but it is still less error prone
	// than manual computation
	LinearLayer validator("", kernel, bias);

	CMatrix cInputErrors = CMatrix::Zero(25, 1);
	CMap cMapInputErrors(cInputErrors.data(), 5, 5);
	CMap mapInput(input.data(), 5, 5);

	for (int y = 0; y < 3; y += 2)
	{
		for (int x = 0; x < 3; x += 2)
		{
			RMatrix bInWindow = mapInput.block(y, x, 3, 3);

			CMatrix inWindow = RMap(bInWindow.data(), 9, 1);

			CMatrix opErrWindow = outputErrors.block(y * 3 + x, 0, 1, 1);

			Params linInputErrors = validator.UTBackprop(0,
													   inWindow,
													   opErrWindow);

			CMatrix linErrWindow = RMap(linInputErrors.Data.data(), 3, 3);

			cMapInputErrors.block(y, x, 3, 3) += linErrWindow;
		}
	}

	AssertMatrixEquivalence(inputErrors.Data, cInputErrors);
}

TEST(ConvoLayerTest, PaddedBackprop)
{
	RMatrix kernel(1, 9);
	kernel << 1, 1, 1,
			  1, 1, 1,
			  1, 1, 1;
	Vector bias(1);
	bias << 1;

	CMatrix input(1, 1);
	input << 4;

	CMatrix outputErrors(1, 1);
	outputErrors << 4;

	RMatrix weightsGrad;
	Vector biasGrad;

	Params inputErrors = Backprop(kernel, bias,
								  Params(1, 1, 1, input),
								  Params(1, 1, 1, outputErrors),
								  3, 3,
								  1, 1,
								  1, 1,
								  weightsGrad, biasGrad);

	LinearLayer validator("", kernel, bias);

	CMatrix linput(9, 1);
	linput << 0, 0, 0,
			  0, 4, 0,
			  0, 0, 0;

	Params linInputErrors = validator.UTBackprop(0,
											   linput,
											   outputErrors);

	AssertMatrixEquivalence(inputErrors.Data,
							linInputErrors.Data.block(5, 0, 1, 1));

	AssertMatrixEquivalence(validator.GetParams(0).WeightsGrad,
							weightsGrad);

	AssertVectorEquivalence(validator.GetParams(0).BiasGrad,
							biasGrad);
}



















