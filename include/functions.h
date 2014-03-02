#pragma once

#include "math_util.h"

struct NEURAL_NET_API LinearFn
{
	static std::string Type() {
		return "Linear";
	}

	static const bool NEEDS_INPUT = false;

	static Real Compute(Real input);
	static Real Derivative(Real input, Real lastOut);
};

struct NEURAL_NET_API LogisticFn
{
	static std::string Type()
	{
		return "Logistic";
	}

	static const bool NEEDS_INPUT = false;

	static Real Compute(Real input);
	static Real Derivative(Real input, Real lastOut);
};

struct NEURAL_NET_API RectifierFn
{
	static std::string Type()
	{
		return "Rectifier";
	}

	static const bool NEEDS_INPUT = false;

	static Real Compute(Real input);
	static Real Derivative(Real input, Real lastOut);
};

struct NEURAL_NET_API TanhFn
{
	static std::string Type()
	{
		return "Tanh";
	}

	static const bool NEEDS_INPUT = LogisticFn::NEEDS_INPUT;

	static Real Compute(Real input);
	static Real Derivative(Real input, Real lastOut);
};

struct NEURAL_NET_API RampFn
{
	static std::string Type()
	{
		return "Ramp";
	}

	static const bool NEEDS_INPUT = false;

	static Real Compute(Real input);
	static Real Derivative(Real input, Real lastOut);
};