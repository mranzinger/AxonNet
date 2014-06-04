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
      HostMat(nullptr), CudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(CMatrix *hostMat)
    : Width(1), Height(hostMat->rows()), Depth(1),
      Rows(hostMat->rows()), Cols(hostMat->cols()),
      HostMat(hostMat), CudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(CuMat *cudaMat)
    : Width(1), Height(CuMat_Rows(cudaMat)), Depth(1),
      Rows(CuMat_Rows(cudaMat)), Cols(CuMat_Cols(cudaMat)),
      HostMat(nullptr), CudaMat(cudaMat)
{
    _refCt = new uint32_t(1);
}
Params::Params(size_t width, size_t height, size_t depth,
               CMatrix *hostMat)
    : Width(width), Height(height), Depth(depth),
      Rows(hostMat->rows()), Cols(hostMat->cols()),
      HostMat(hostMat), CudaMat(nullptr)
{
    _refCt = new uint32_t(1);
}
Params::Params(size_t width, size_t height, size_t depth,
               CuMat *cudaMat)
    : Width(width), Height(height), Depth(depth),
      Rows(CuMat_Rows(cudaMat)), Cols(CuMat_Cols(cudaMat)),
      HostMat(nullptr), CudaMat(cudaMat)
{
    _refCt = new uint32_t(1);
}
Params::Params(const Params &other)
    : Width(other.Width),
      Height(other.Height),
      Depth(other.Depth),
      Rows(other.Rows),
      Cols(other.Cols),
      HostMat(other.HostMat),
      CudaMat(other.CudaMat),
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
      HostMat(hostMat),
      CudaMat(nullptr)
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
      HostMat(nullptr),
      CudaMat(cudaMat)
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
        delete HostMat;
        CuMat_Delete(CudaMat);
        delete _refCt;
    }
}

Params &Params::operator=(Params other)
{
    swap(*this, other);
    return *this;
}

void swap(Params &a, Params &b)
{
    swap(a.Width, b.Width);
    swap(a.Height, b.Height);
    swap(a.Depth, b.Depth);
    swap(a._refCt, b._refCt);
    swap(a.HostMat, b.HostMat);
    swap(a.CudaMat, b.CudaMat);
}
