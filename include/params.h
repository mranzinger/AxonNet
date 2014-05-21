#pragma once

#include <map>
#include <string>

#include "math_util.h"

struct NEURAL_NET_API Params
{
public:
	enum LayoutKind { Packed, Planar };

	size_t Width;
	size_t Height;
	size_t Depth;

	LayoutKind Layout;

	/*
	 * Input data matrix. Supports mini-batch when the number of
	 * columns > 1. Data is stored column major, so accessing the kth element
	 * of the ith column is (i * #rows) + k
	 */
	CMatrix Data;

	Params() : Width(0), Height(0), Depth(0), Layout(Packed) { }
	Params(size_t width, size_t height, size_t depth, CMatrix data, LayoutKind layout = Packed)
		: Width(width), Height(height), Depth(depth), Layout(layout)
	{
		Data.swap(data);
	}
	Params(CMatrix data)
		: Width(data.rows()), Height(1), Depth(1), Layout(Packed)
	{
		Data.swap(data);
	}
	Params(const Params &other)
		: Width(other.Width), Height(other.Height), Depth(other.Depth),
		  Layout(other.Layout), Data(other.Data) 
	{ 
	}
	Params(const Params &other, CMatrix data)
		: Width(other.Width), Height(other.Height), Depth(other.Depth),
		  Layout(other.Layout)
	{
		Data.swap(data);
	}
	Params(Params &&other)
	{
		swap(*this, other);
	}

	Params &operator=(Params other)
	{
		swap(*this, other);
		return *this;
	}
	Params &operator=(CMatrix data)
	{
		Width = data.size();
		Height = 1;
		Depth = 1;
		Layout = Packed;
		Data.swap(data);
		return *this;
	}

	bool operator==(const Params &other) const
	{
		return Width == other.Width && Height == other.Height && Depth == other.Depth &&
			Data == other.Data;
	}
	bool operator!=(const Params &other) const
	{
		return !(*this == other);
	}

	size_t size() const
	{
		return Width * Height * Depth;
	}

	size_t BatchSize() const { return Data.cols(); }

	friend void swap(Params &a, Params &b)
	{
		using std::swap;

		swap(a.Width, b.Width);
		swap(a.Height, b.Height);
		swap(a.Depth, b.Depth);
		swap(a.Layout, b.Layout);
		a.Data.swap(b.Data);
	}
};

typedef std::vector<Params> MultiParams;
typedef std::map<std::string, Params> ParamMap;

