/*
 * File description: params.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "params.h"
#include "cumat_host_helper.h"

using namespace std;

Params::Params()
    : Width(0), Height(0), Depth(0),
      Rows(0), Cols(0),
      _hostMat(nullptr), _cudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(CMatrix *hostMat)
    : Width(1), Height(hostMat->rows()), Depth(1),
      Rows(hostMat->rows()), Cols(hostMat->cols()),
      _hostMat(hostMat), _cudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(CuMat *cudaMat)
    : Width(1), Height(CuMat_Rows(cudaMat)), Depth(1),
      Rows(CuMat_Rows(cudaMat)), Cols(CuMat_Cols(cudaMat)),
      _hostMat(nullptr), _cudaMat(cudaMat)
{
    _refCt = new uint32_t(1);
}
Params::Params(size_t width, size_t height, size_t depth,
               CMatrix *hostMat)
    : Width(width), Height(height), Depth(depth),
      Rows(hostMat->rows()), Cols(hostMat->cols()),
      _hostMat(hostMat), _cudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(size_t width, size_t height, size_t depth,
               CuMat *cudaMat)
    : Width(width), Height(height), Depth(depth),
      Rows(CuMat_Rows(cudaMat)), Cols(CuMat_Cols(cudaMat)),
      _hostMat(nullptr), _cudaMat(cudaMat)
{
    _refCt = new uint32_t(1);
}
Params::Params(const Params &other)
    : Width(other.Width),
      Height(other.Height),
      Depth(other.Depth),
      Rows(other.Rows),
      Cols(other.Cols),
      _hostMat(other._hostMat),
      _cudaMat(other._cudaMat),
      _refCt(other._refCt)
{
    // Increment the reference count to this data
    ++*_refCt;
}
Params::Params(const Params &like, CMatrix *hostMat)
    : Width(like.Width),
      Height(like.Height),
      Depth(like.Depth),
      Rows(like.Rows),
      Cols(like.Cols),
      _hostMat(hostMat),
      _cudaMat(nullptr)
{
    assert(Rows == hostMat->rows());
    assert(Cols == hostMat->cols());

    _refCt = new uint32_t(1);
}
Params::Params(const Params &like, CuMat *cudaMat)
    : Width(like.Width),
      Height(like.Height),
      Depth(like.Depth),
      Rows(like.Rows),
      Cols(like.Cols),
      _hostMat(nullptr),
      _cudaMat(cudaMat)
{
    assert(Rows == CuMat_Rows(cudaMat));
    assert(Cols == CuMat_Cols(cudaMat));

    _refCt = new uint32_t(1);
}
Params::Params(Params &&other)
    : Params()
{
    swap(*this, other);
}

Params::~Params()
{
    --*_refCt;
    if (*_refCt == 0)
    {
        delete _hostMat;
        CuMat_Delete(_cudaMat);
        delete _refCt;
    }
}

Params &Params::operator=(Params other)
{
    swap(*this, other);
    return *this;
}

const CMatrix &Params::GetHostMatrix() const
{
    return const_cast<Params*>(this)->GetHostMatrix();
}

CMatrix &Params::GetHostMatrix()
{
    if (!_hostMat)
    {
        assert(_cudaMat);
        _hostMat = CuMat_CopyToHost(*_cudaMat);
    }
    return *_hostMat;
}

const CuMat &Params::GetCudaMatrix(CuContext handle) const
{
    return const_cast<Params*>(this)->GetCudaMatrix(handle);
}

CuMat &Params::GetCudaMatrix(CuContext handle)
{
    if (!_cudaMat)
    {
        assert(_hostMat);
        assert(handle.CublasHandle);
        _cudaMat = CuMat_CopyToDevice(*_hostMat, handle);
    }
    return *_cudaMat;
}

void swap(Params &a, Params &b)
{
    swap(a.Width, b.Width);
    swap(a.Height, b.Height);
    swap(a.Depth, b.Depth);
    swap(a._refCt, b._refCt);
    swap(a._hostMat, b._hostMat);
    swap(a._cudaMat, b._cudaMat);
}

Params Params::CreateLike(const Params& other)
{
	if (other._hostMat)
	{
		return Params(other, new CMatrix(other.Rows, other.Cols));
	}
	else if (other._cudaMat)
	{
		return Params(other, CuMat_MakeSimilar(*other._cudaMat));
	}
	else
		throw runtime_error("Invalid data buffer");
}

Params Params::CreateLike(const Params& other, const CuContext& handle)
{
	return Params(other, CuMat_Make(handle, other.Rows, other.Cols));
}
