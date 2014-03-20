#pragma once

#include <immintrin.h>

#include <Eigen/Dense>

template<size_t Alignment, typename T>
__forceinline T *GetNextAligned(T *pVal)
{
	return (T*) ((uintptr_t(pVal) & ~(Alignment-1)) + Alignment);
}

template<size_t Alignment, typename T>
__forceinline bool IsAlignedToBoundary(T *pVal)
{
	return (uintptr_t(pVal) & (Alignment - 1)) == 0;
}

#define FM_AVX
#ifdef FM_AVX
__forceinline void ScalarMultiplyStragglers(float *&pCurr, float *pEnd, float val)
{
	ptrdiff_t leftover = pEnd - pCurr;
	switch (leftover)
	{
	case 7:
		*pCurr++ = val * *pCurr;
	case 6:
		*pCurr++ = val * *pCurr;
	case 5:
		*pCurr++ = val * *pCurr;
	case 4:
		*pCurr++ = val * *pCurr;
	case 3:
		*pCurr++ = val * *pCurr;
	case 2:
		*pCurr++ = val * *pCurr;
	case 1:
		*pCurr++ = val * *pCurr;
	case 0:
		break;
	}
}

template<typename MatType>
__forceinline void ScalarMultiply(MatType &mat, float val)
{
	float *pData = mat.data();
	float *pEnd = pData + mat.size();

	float *pCurr = pData;

	float *pAvxStart = GetNextAligned<32>(pData);

	// Do the simple routine for the unaligned start of the data
	ScalarMultiplyStragglers(pCurr, pAvxStart, val);

	// Truncate the stragglers
	float *pAvxEnd = pCurr + ((pEnd - pCurr) & ~0x7);

	__m256 vVal = _mm256_set1_ps(val);

	for (; pCurr != pAvxEnd; pCurr += 8)
	{
		_mm256_store_ps(pCurr, _mm256_mul_ps(vVal, _mm256_load_ps(pCurr)));
	}

	ScalarMultiplyStragglers(pCurr, pEnd, val);
}

__forceinline void AddStragglers(float *&pA, const float *&pB, float *pEnd)
{
	ptrdiff_t leftover = pEnd - pA;
	switch (leftover)
	{
	case 7:
		*pA++ += *pB++;
	case 6:
		*pA++ += *pB++;
	case 5:
		*pA++ += *pB++;
	case 4:
		*pA++ += *pB++;
	case 3:
		*pA++ += *pB++;
	case 2:
		*pA++ += *pB++;
	case 1:
		*pA++ += *pB++;
	case 0:
		break;
	};
}

__forceinline void AddScaledStragglers(float *&pA, const float *&pB, float *pEnd, float otherScale)
{
	ptrdiff_t leftover = pEnd - pA;
	switch (leftover)
	{
	case 7:
		*pA++ += otherScale * *pB++;
	case 6:
		*pA++ += otherScale * *pB++;
	case 5:
		*pA++ += otherScale * *pB++;
	case 4:
		*pA++ += otherScale * *pB++;
	case 3:
		*pA++ += otherScale * *pB++;
	case 2:
		*pA++ += otherScale * *pB++;
	case 1:
		*pA++ += otherScale * *pB++;
	case 0:
		break;
	};
}

template<bool AAligned, bool BAligned, bool Scaled = true>
__forceinline void AddScaled(float *pMat, const float *pOther, float *pEnd, float otherScale)
{
	// If neither are aligned, then we can clean up the stragglers to a boundary
	// and finish aligned
	if (!(AAligned || BAligned))
	{
		float *pMatAligned = GetNextAligned<32>(pMat);

		if (Scaled)
			AddScaledStragglers(pMat, pOther, pMatAligned, otherScale);
		else
			AddStragglers(pMat, pOther, pMatAligned);

		AddScaled<true, true, Scaled>(pMat, pOther, pEnd, otherScale);
		return;
	}

	float *pAvxEnd = pMat + ((pEnd - pMat) & ~0x7);

	__m256 vScale = _mm256_set1_ps(otherScale);

	for (; pMat != pAvxEnd; pMat += 8, pOther += 8)
	{
		const __m256 vMat = AAligned ? _mm256_load_ps(pMat) : _mm256_loadu_ps(pMat);
		const __m256 vOther = BAligned ? _mm256_load_ps(pOther) : _mm256_loadu_ps(pOther);

		const __m256 vResult = Scaled ?
								_mm256_fmadd_ps(vScale, vOther, vMat) // a * b + c
								:
								_mm256_add_ps(vMat, vOther); // a + b

		if (AAligned)
			_mm256_store_ps(pMat, vResult);
		else
			_mm256_storeu_ps(pMat, vResult);
	}

	if (Scaled)
		AddScaledStragglers(pMat, pOther, pEnd, otherScale);
	else
		AddStragglers(pMat, pOther, pEnd);
}

template<bool Scaled, typename MatType>
__forceinline void AddScaledImpl(MatType &mat, const MatType &other, float otherScale)
{
	float *pMat = mat.data();
	const float *pOther = other.data();
	float *pEnd = pMat + mat.size();

	// Need to figure out the optimal way to compute this since mat and other
	// may have different alignments. Mixed alignments is worst-case, with mat
	// being unaligned and other aligned the absolute worst
	bool matAligned = IsAlignedToBoundary<32>(pMat);
	bool otherAligned = IsAlignedToBoundary<32>(pOther);

	if (matAligned && otherAligned)
		AddScaled<true, true, Scaled>(pMat, pOther, pEnd, otherScale);
	else if (!(matAligned || otherAligned))
		AddScaled<false, false, Scaled>(pMat, pOther, pEnd, otherScale);
	else if (matAligned)
		AddScaled<true, false, Scaled>(pMat, pOther, pEnd, otherScale);
	else
		AddScaled<false, true, Scaled>(pMat, pOther, pEnd, otherScale);
}
#else
__forceinline void ScalarMultiplyStragglers(float *&pCurr, float *pEnd, float val)
{
	ptrdiff_t leftover = pEnd - pCurr;
	switch (leftover)
	{
	case 3:
		*pCurr++ = val * *pCurr;
	case 2:
		*pCurr++ = val * *pCurr;
	case 1:
		*pCurr++ = val * *pCurr;
	case 0:
		break;
	}
}

template<typename MatType>
__forceinline void ScalarMultiply(MatType &mat, float val)
{
	float *pData = mat.data();
	float *pEnd = pData + mat.size();

	float *pCurr = pData;

	// Truncate the stragglers
	float *pAvxEnd = pCurr + ((pEnd - pCurr) & ~0x7);

	__m128 vVal = _mm_set1_ps(val);

	for (; pCurr != pAvxEnd; pCurr += 4)
	{
		_mm_store_ps(pCurr, _mm_mul_ps(vVal, _mm_load_ps(pCurr)));
	}

	ScalarMultiplyStragglers(pCurr, pEnd, val);
}

__forceinline void AddStragglers(float *&pA, const float *&pB, float *pEnd)
{
	ptrdiff_t leftover = pEnd - pA;
	switch (leftover)
	{
	case 3:
		*pA++ += *pB++;
	case 2:
		*pA++ += *pB++;
	case 1:
		*pA++ += *pB++;
	case 0:
		break;
	};
}

__forceinline void AddScaledStragglers(float *&pA, const float *&pB, float *pEnd, float otherScale)
{
	ptrdiff_t leftover = pEnd - pA;
	switch (leftover)
	{
	case 3:
		*pA++ += otherScale * *pB++;
	case 2:
		*pA++ += otherScale * *pB++;
	case 1:
		*pA++ += otherScale * *pB++;
	case 0:
		break;
	};
}

template<bool Scaled = true>
__forceinline void AddScaled(float *pMat, const float *pOther, float *pEnd, float otherScale)
{
	float *pAvxEnd = pMat + ((pEnd - pMat) & ~0x7);

	__m128 vScale = _mm_set1_ps(otherScale);

	for (; pMat != pAvxEnd; pMat += 4, pOther += 4)
	{
		const __m128 vMat = _mm_load_ps(pMat);
		const __m128 vOther = _mm_load_ps(pOther);

		const __m128 vResult = Scaled ?
								_mm_fmadd_ps(vScale, vOther, vMat) // a * b + c
								:
								_mm_add_ps(vMat, vOther); // a + b

		_mm_store_ps(pMat, vResult);
	}

	if (Scaled)
		AddScaledStragglers(pMat, pOther, pEnd, otherScale);
	else
		AddStragglers(pMat, pOther, pEnd);
}

template<bool Scaled, typename MatType>
__forceinline void AddScaledImpl(MatType &mat, const MatType &other, float otherScale)
{
	AddScaled<Scaled>(mat.data(), other.data(), mat.data() + mat.size(), otherScale);
}
#endif

template<typename MatType>
__forceinline void AddScaled(MatType &mat, const MatType &other, float otherScale)
{
	AddScaledImpl<true>(mat, other, otherScale);
}

template<typename MatType>
__forceinline void Add(MatType &mat, const MatType &other)
{
	AddScaledImpl<false>(mat, other, 1.0f);
}

template<typename MatType>
__forceinline void SetZero(MatType &mat)
{
	memset(mat.data(), 0, sizeof(Real) * mat.size());
}