#include "functions.h"
#include <xmmintrin.h>
#include "sse_mathfun.h"

using namespace std;

Real LinearFn::Compute(Real input)
{
	return input;
}

CMatrix LinearFn::Compute(const CMatrix &input)
{
	return input;
}

Real LinearFn::Derivative(Real input)
{
	return 1;
}

CMatrix LinearFn::Derivative(const CMatrix &input)
{
	return CMatrix(input.rows(), input.cols()).setOnes();
}

Real LogisticFn::Compute(Real input)
{
	return 1.0f / (1.0f + exp(-input));
}

CMatrix LogisticFn::Compute(const CMatrix &input)
{
	const __m128 s_one = _mm_set1_ps(1);
	const __m128 s_zero = _mm_set1_ps(0);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const __m128 ipVal = _mm_load_ps(pInput);

		const __m128 deno = _mm_add_ps(s_one, exp_ps(_mm_sub_ps(s_zero, ipVal)));

		const __m128 opVal = _mm_mul_ps(s_one, _mm_rcp_ps(deno));

		_mm_store_ps(pOutput, opVal);
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Compute(*pInput);
	}

	return move(ret);
}

Real LogisticFn::Derivative(Real input)
{
	return Derivative(input, Compute(input));
}

Real LogisticFn::Derivative(Real input, Real computeOutput)
{
	return computeOutput * (1.0f - computeOutput);
}

CMatrix LogisticFn::Derivative(const CMatrix &input)
{
	return Derivative(input, Compute(input));
}

CMatrix LogisticFn::Derivative(const CMatrix &input, CMatrix computeOutput)
{
	const __m128 s_one = _mm_set1_ps(1);

	float *pVals = computeOutput.data();
	float *pSSEEnd = pVals + (input.size() & ~0x3);

	for (; pVals != pSSEEnd; pVals += 4)
	{
		const __m128 ipVal = _mm_load_ps(pVals);

		const __m128 opVal = _mm_mul_ps(ipVal, _mm_sub_ps(s_one, ipVal));

		_mm_store_ps(pVals, opVal);
	}

	for (const float *pEnd = computeOutput.data() + computeOutput.size(); pVals != pEnd; ++pVals)
	{
		*pVals = Derivative(0.0f, *pVals);
	}

	return computeOutput;
}

Real RectifierFn::Compute(Real input)
{
	return input > 0 ? input : 0;
}

CMatrix RectifierFn::Compute(const CMatrix &input)
{
	const __m128 s_zero = _mm_set1_ps(0);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const auto val = _mm_load_ps(pInput);

		_mm_store_ps(pOutput, _mm_max_ps(s_zero, val));
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Compute(*pInput);
	}

	return move(ret);
}

Real RectifierFn::Derivative(Real input)
{
	return input > 0 ? 1 : 0;
}

CMatrix RectifierFn::Derivative(const CMatrix &input)
{
	const __m128 s_one = _mm_set1_ps(1);
	const __m128 s_zero = _mm_set1_ps(0);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const __m128 val = _mm_load_ps(pInput);

		const __m128 gt0 = _mm_cmpgt_ps(val, s_zero);

		const __m128 rVal = _mm_or_ps(_mm_and_ps(gt0, s_one), _mm_andnot_ps(gt0, s_zero));

		_mm_store_ps(pOutput, rVal);
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Derivative(*pInput);
	}

	return move(ret);
}

Real HardTanhFn::Compute(Real input)
{
	if (input < -1)
		return -1;
	else if (input > 1)
		return 1;
	return input;
}

CMatrix HardTanhFn::Compute(const CMatrix &input)
{
	const __m128 s_neg1 = _mm_set1_ps(-1);
	const __m128 s_1 = _mm_set1_ps(1);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const auto val = _mm_load_ps(pInput);

		_mm_store_ps(pOutput, _mm_min_ps(s_1, _mm_max_ps(s_neg1, val)));
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Compute(*pInput);
	}

	return move(ret);
}

Real HardTanhFn::Derivative(Real input)
{
	if (input < -1 || input > 1)
		return 0;
	return 1;
}

CMatrix HardTanhFn::Derivative(const CMatrix &input)
{
	const __m128 s_neg1 = _mm_set1_ps(-1);
	const __m128 s_1 = _mm_set1_ps(1);
	const __m128 s_0 = _mm_set1_ps(0);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const auto val = _mm_load_ps(pInput);

		const __m128 valid = _mm_and_ps(_mm_cmpgt_ps(val, s_neg1), _mm_cmplt_ps(val, s_1));

		const __m128 rVal = _mm_or_ps(_mm_and_ps(valid, s_1), _mm_andnot_ps(valid, s_0));

		_mm_store_ps(pOutput, rVal);
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Derivative(*pInput);
	}

	return move(ret);
}



Real SoftPlusFn::Compute(Real input)
{
	if (input > 20)
		return input;

	return log(1 + exp(input));
}

CMatrix SoftPlusFn::Compute(const CMatrix &input)
{
	static const __m128 s_1 = _mm_set1_ps(1);
	static const __m128 s_20 = _mm_set1_ps(20);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const __m128 val = _mm_load_ps(pInput);

		const __m128 gt20 = _mm_cmpgt_ps(val, s_20);

		const __m128 soft = log_ps(_mm_add_ps(s_1, exp_ps(val)));

		const __m128 result = _mm_or_ps(_mm_and_ps(gt20, val), _mm_andnot_ps(gt20, soft));

		_mm_store_ps(pOutput, result);
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Compute(*pInput);
	}

	return move(ret);
}

Real SoftPlusFn::Derivative(Real input)
{
	return LogisticFn::Compute(input);
}

CMatrix SoftPlusFn::Derivative(const CMatrix &input)
{
	return LogisticFn::Compute(input);
}

Real TanhFn::Compute(Real input)
{
	Real e = exp(-input);

	return (1.0f - e) / (1.0 + e);
}

CMatrix TanhFn::Compute(const CMatrix &input)
{
	static const __m128 s_0 = _mm_set1_ps(0);
	static const __m128 s_1 = _mm_set1_ps(1);

	CMatrix ret(input.rows(), input.cols());

	const float *pInput = input.data();
	float *pOutput = ret.data();

	const float *pAvxEnd = pInput + (input.size() & ~0x3);

	for (; pInput != pAvxEnd; pInput += 4, pOutput += 4)
	{
		const __m128 val = _mm_load_ps(pInput);

		const __m128 e = exp_ps(_mm_sub_ps(s_0, val));

		const __m128 result = _mm_div_ps(
								_mm_sub_ps(s_1, e),
								_mm_add_ps(s_1, e)
							  );

		_mm_store_ps(pOutput, result);
	}

	for (const float *pEnd = input.data() + input.size(); pInput != pEnd; ++pInput, ++pOutput)
	{
		*pOutput = Compute(*pInput);
	}

	return move(ret);
}

Real TanhFn::Derivative(Real input)
{
	return Derivative(input, Compute(input));
}

Real TanhFn::Derivative(Real input, Real computeOutput)
{
	return 1.0 - Square(computeOutput);
}

CMatrix TanhFn::Derivative(const CMatrix &input)
{
	return Derivative(input, Compute(input));
}

CMatrix TanhFn::Derivative(const CMatrix &input, CMatrix computeOutput)
{
	static const __m128 s_one = _mm_set1_ps(1);

	float *pVals = computeOutput.data();
	float *pSSEEnd = pVals + (input.size() & ~0x3);

	for (; pVals != pSSEEnd; pVals += 4)
	{
		const __m128 ipVal = _mm_load_ps(pVals);

		const __m128 opVal = _mm_sub_ps(s_one, _mm_mul_ps(ipVal, ipVal));

		_mm_store_ps(pVals, opVal);
	}

	for (const float *pEnd = computeOutput.data() + computeOutput.size(); pVals != pEnd; ++pVals)
	{
		*pVals = Derivative(0.0f, *pVals);
	}

	return move(computeOutput);
}

Real RampFn::Compute(Real input)
{
	if (input < -2)
		return -1;
	if (input > 2)
		return 1;

	return .5 * input;
}

Real RampFn::Derivative(Real input)
{
	if (input < -2 || input > 2)
		return 0;
	return .5;
}
