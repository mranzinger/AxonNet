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
	Real val = Compute(input);

	return val * (1 - val);
}

Real RectifierFn::Compute(Real input)
{
	return input > 0 ? input : 0;
}

Real RectifierFn::Derivative(Real input)
{
	return input >= 0 ? 1 : 0;
}

Real TanhFn::Compute(Real input)
{
	return tanh(input);
}

Real TanhFn::Derivative(Real input)
{
	Real val = Compute(input);

	return 1 - Square(val);
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

Real SoftPlusFn::Compute(Real input)
{
	return log(1 + exp(input));
}

Real SoftPlusFn::Derivative(Real input)
{
	return LogisticFn::Compute(input);
}

Real HardTanhFn::Compute(Real input)
{
	if (input < -1)
		return -1;
	else if (input > 1)
		return 1;
	return input;
}

Real HardTanhFn::Derivative(Real input)
{
	if (input < -1 || input > 1)
		return 0;
	return 1;
}