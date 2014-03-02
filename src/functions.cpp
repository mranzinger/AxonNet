#include "functions.h"

Real LinearFn::Compute(Real input)
{
	return input;
}

Real LinearFn::Derivative(Real input, Real lastOut)
{
	return 1;
}

Real LogisticFn::Compute(Real input)
{
	return 1.0f / (1.0f + exp(-input));
}

Real LogisticFn::Derivative(Real input, Real lastOut)
{
	return lastOut * (1 - lastOut);
}

Real RectifierFn::Compute(Real input)
{
	return input > 0 ? input : 0;
}

Real RectifierFn::Derivative(Real input, Real lastOut)
{
	return lastOut >= 0 ? 1 : 0;
}

Real TanhFn::Compute(Real input)
{
	return 2 * LogisticFn::Compute(input) - 1;
}

Real TanhFn::Derivative(Real input, Real lastOut)
{
	return 2 * LogisticFn::Derivative(input, lastOut);
}

Real RampFn::Compute(Real input)
{
	if (input < -2)
		return -1;
	if (input > 2)
		return 1;

	return .5 * input;
}

Real RampFn::Derivative(Real input, Real lastOut)
{
	if (lastOut == -1 || lastOut == 1)
		return 0;
	return .5;
}

Real SoftPlusFn::Compute(Real input)
{
	return log(1 + exp(input));
}

Real SoftPlusFn::Derivative(Real input, Real lastOut)
{
	return LogisticFn::Compute(input);
}