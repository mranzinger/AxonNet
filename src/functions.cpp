#include "functions.h"

Real LinearFn::Compute(Real input)
{
	return input;
}

Real LinearFn::Derivative(Real input)
{
	return 1;
}

Real LogisticFn::Compute(Real input)
{
	return 1.0f / (1.0f + exp(-input));
}

Real LogisticFn::Derivative(Real input)
{
	Real v = Compute(input);

	return v * (1 - v);
}

Real RectifierFn::Compute(Real input)
{
	return input > 0 ? input : 0;
}

Real RectifierFn::Derivative(Real input)
{
	return input > 0 ? 1 : 0;
}

Real TanhFn::Compute(Real input)
{
	return 2 * LogisticFn::Compute(input) - 1;
}

Real TanhFn::Derivative(Real input)
{
	return 2 * LogisticFn::Derivative(input);
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
	if (input <= -2 || input >= 2)
		return 0;
	return .5;
}