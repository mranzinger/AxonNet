/*
 * File description: memset_util.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#ifndef MEMSET_UTIL_H_
#define MEMSET_UTIL_H_

template<typename T>
void memset32(void *dest, T value, uintptr_t size)
{
	static_assert(sizeof(T) == 4, "The type of T must be represented in 32-bits");

	char *cDest = (char*)dest;
	char *cEnd = cDest + size * 4;

	for (; cDest != cEnd; cDest += 4)
	{
		memcpy(cDest, &value, 4);
	}
}

template<typename T>
void memsetMany(void *dest, T *pattern, uintptr_t patternSize, uintptr_t destSize)
{
	size_t patternSzBts = patternSize * sizeof(T);

	char *cDest = (char*)dest;
	char *cEnd = cDest + destSize * patternSzBts;

	for (; cDest != cEnd; cDest += patternSzBts)
	{
		memcpy(cDest, pattern, patternSzBts);
	}
}


#endif /* MEMSET_UTIL_H_ */
