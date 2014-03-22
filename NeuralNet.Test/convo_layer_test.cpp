#include "stdafx.h"
#include "CppUnitTest.h"
#include "convo_layer.h"
#include "lin_test.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetTest
{
	TEST_CLASS(convo_layer_test)
	{
	private:
		Params Compute(const Matrix &kernel, const Vector &bias, const Params &input,
			size_t windowSizeX, size_t windowSizeY,
			size_t strideX, size_t strideY,
			ConvoLayer::PaddingMode padMode = ConvoLayer::NoPadding,
			bool isTraining = false)
		{
			return ConvoLayer("", kernel, bias, windowSizeX, windowSizeY,
				strideX, strideY, padMode).Compute(0, input, isTraining);
		}

	public:
		
		TEST_METHOD(SimpleCompute)
		{
			Matrix kernel(1, 9);
			kernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
			Vector bias(1);
			bias << 0;

			Vector input(9);
			input << 1, 1, 1, 1, 1, 1, 1, 1, 1;

			Vector correctOutput(1);
			correctOutput << 0;

			Params comp = Compute(kernel, bias, Params(3, 3, 1, input),
									3, 3, 
									1, 1);

			AssertVectorEquivalence(correctOutput, comp.Data);
		}

		TEST_METHOD(SimpleCompute2)
		{
			Matrix kernel(1, 9);
			kernel << 1, 1, 1, 
					  1, 1, 1, 
					  1, 1, 1;
			Vector bias(1);
			bias << -9;

			Vector input(16);
			input << 1, 2, 3, 4,
					 1, 2, 3, 4,
					 1, 2, 3, 4,
					 1, 2, 3, 4;

			// Output:
			// 18 - 9   27 - 9
			// 18 - 9   27 - 9
			Vector correctOutput(4);
			correctOutput << 9, 18, 9, 18;

			Params comp = Compute(kernel, bias, Params(4, 4, 1, input),
								  3, 3,
								  1, 1);

			AssertVectorEquivalence(correctOutput, comp.Data);
		}

		TEST_METHOD(StrideCompute)
		{
			Matrix kernel(1, 9);
			kernel << 1, 1, 1,
					  1, 1, 1,
					  1, 1, 1;
			Vector bias(1);
			bias << 0;

			Vector input(25);
			input << 1, 2, 3, 4, 5,
					 1, 2, 3, 4, 5,
					 1, 2, 3, 4, 5,
					 1, 2, 3, 4, 5,
					 1, 2, 3, 4, 5;

			// Output:
			// 18  36
			// 18  36
			Vector correctOutput(4);
			correctOutput << 18, 36, 18, 36;

			Params comp = Compute(kernel, bias, Params(5, 5, 1, input),
								  3, 3,
								  2, 2);

			AssertVectorEquivalence(correctOutput, comp.Data);
		}

	};
}