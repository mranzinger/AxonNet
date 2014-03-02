#pragma once

#include "math_util.h"

struct NEURAL_NET_API LinearFn
{
	static std::string Type() {
		return "Linear";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API LogisticFn
{
	static std::string Type()
	{
		return "Logistic";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API RectifierFn
{
	static std::string Type()
	{
		return "Rectifier";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API TanhFn
{
	static std::string Type()
	{
		return "Tanh";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API RampFn
{
	static std::string Type()
	{
		return "Ramp";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};