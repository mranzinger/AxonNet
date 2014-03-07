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

struct NEURAL_NET_API SoftPlusFn
{
	static std::string Type()
	{
		return "SoftPlus";
	}

	static Real Compute(Real input);
	static Real Derivative(Real input);
};

struct NEURAL_NET_API HardTanhFn
{
	static std::string Type()
	{
		return "HardTanh";
	}

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
	template<typename Fn, bool IsExplicit>
	struct FnDvApplicator
	{
		static Vector Apply(const Vector &input)
		{
			return input.unaryExpr([](Real val) { return Fn::Derivative(val); });
		}
	};

	template<typename Fn>
	struct FnApplicator<Fn, true>
	{
		static Vector Apply(const Vector &input)
		{
			return Fn::VecCompute(input);
		}
	};

	template<typename Fn>
	struct FnDvApplicator<Fn, true>
	{
		static Vector Apply(const Vector &input)
		{
			return Fn::VecDerivative(input);
		}
	};

	template<typename Fn>
	struct has_vec_compute
	{
	private:
		struct no {
			char[2];
		};

		template<typename C> static char test(char[sizeof(&C::VecCompute)]);
		template<typename C> static no   test(...);

	public:
		enum { value = sizeof(test<Fn>(0)) == sizeof(char) };
	};

	template<typename Fn>
	struct has_vec_derivative
	{
	private:
		struct no {
			char[2];
		};

		template<typename C> static char test(char[sizeof(&C::VecDerivative)]);
		template<typename C> static no   test(...);

	public:
		enum { value = sizeof(test<Fn>(0)) == sizeof(char) };
	};
}

template<typename Fn>
Vector ApplyFunction(const Vector &input)
{
	return FnApplicator<Fn, has_vec_compute<Fn>::value>::Apply(input);
}

template<typename Fn>
Vector ApplyDerivative(const Vector &input)
{
	return FnDvApplicator<Fn, has_vec_derivative<Fn>::value>::Apply(input);
}