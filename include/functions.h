#pragma once

#include "math_util.h"

struct NEURAL_NET_API LinearFn
{
	static std::string Type() {
		return "Linear";
	}

	static const bool Vectorized = false;
	static const bool Binary = false;

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API LogisticFn
{
	static std::string Type()
	{
		return "Logistic";
	}

	static const bool Vectorized = false;
	static const bool Binary = true;

	static Real Compute(Real input);
	static Real Derivative(Real input);

	static Real Derivative(Real input, Real computeOutput);
};

struct NEURAL_NET_API RectifierFn
{
	static std::string Type()
	{
		return "Rectifier";
	}

	static const bool Vectorized = false;
	static const bool Binary = false;

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API TanhFn
{
	static std::string Type()
	{
		return "Tanh";
	}

	static const bool Vectorized = false;
	static const bool Binary = true;

	static Real Compute(Real input);
	static Real Derivative(Real input);

	static Real Derivative(Real input, Real computeOutput);
};

struct NEURAL_NET_API RampFn
{
	static std::string Type()
	{
		return "Ramp";
	}

	static const bool Vectorized = false;
	static const bool Binary = false;

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API SoftPlusFn
{
	static std::string Type()
	{
		return "SoftPlus";
	}

	static const bool Vectorized = false;
	static const bool Binary = false;

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API HardTanhFn
{
	static std::string Type()
	{
		return "HardTanh";
	}

	static const bool Vectorized = false;
	static const bool Binary = false;

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

namespace 
{
	template<typename Fn, bool IsExplicit>
	struct FnApplicator
	{
		static Vector Apply(const Vector &input)
		{
			return input.unaryExpr([](Real val) { return Fn::Compute(val); });
		}
	};


	template<typename Fn>
	struct FnApplicator<Fn, true>
	{
		static Vector Apply(const Vector &input)
		{
			return Fn::Compute(input);
		}
	};

	// No vectorizing, unary
	template<typename Fn, bool IsExplicit, bool IsBinary>
	struct FnDvApplicator
	{
		static Vector Apply(const Vector &input, const Vector &output)
		{
			return input.unaryExpr([](Real val) { return Fn::Derivative(val); });
		}
	};

	// Vector version, unary
	template<typename Fn>
	struct FnDvApplicator<Fn, true, false>
	{
		static Vector Apply(const Vector &input, const Vector &output)
		{
			return Fn::Derivative(input);
		}
	};

	// No vectorizing, binary
	template<typename Fn>
	struct FnDvApplicator<Fn, false, true>
	{
		static Vector Apply(const Vector &input, const Vector &output)
		{
			return input.binaryExpr(output,
						[] (Real in, Real out)
						{
							return Fn::Derivative(in, out);
						}
			);
		}
	};

	// Vectorized, Binary
	template<typename Fn>
	struct FnDvApplicator<Fn, true, true>
	{
		static Vector Apply(const Vector &input, const Vector &output)
		{
			return Fn::Derivative(input, output);
		}
	};
}

template<typename Fn>
Vector ApplyFunction(const Vector &input)
{
	return FnApplicator<Fn, Fn::Vectorized>::Apply(input);
}

template<typename Fn>
Vector ApplyDerivative(const Vector &input)
{
	static Vector s_dummy;

	return FnDvApplicator<Fn, Fn::Vectorized, false>::Apply(input, s_dummy);
}

template<typename Fn>
Vector ApplyDerivative(const Vector &input, const Vector &computeOutput)
{
	return FnDvApplicator<Fn, Fn::Vectorized, Fn::Binary>::Apply(input, computeOutput);
}
