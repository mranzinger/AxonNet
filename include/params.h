#pragma once

#include "math_util.h"

struct NEURAL_NET_API Params
{
public:
	enum LayoutKind { Packed, Planar };

	size_t Width;
	size_t Height;
	size_t Depth;

	LayoutKind Layout;

	Vector Data;

	Params() : Width(0), Height(0), Depth(0), Layout(Packed) { }
	Params(size_t width, size_t height, size_t depth, Vector data, LayoutKind layout = Packed)
		: Width(width), Height(height), Depth(depth), Layout(layout)
	{
		Data.swap(data);
	}
	Params(Vector data)
		: Width(data.size()), Height(1), Depth(1), Layout(Packed) 
	{
		Data.swap(data);
	}
	Params(const Params &other)
		: Width(other.Width), Height(other.Height), Depth(other.Depth),
		  Layout(other.Layout), Data(other.Data) 
	{ 
	}
	Params(const Params &other, Vector data)
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
	Params &operator=(Vector data)
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

