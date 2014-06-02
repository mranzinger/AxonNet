/*
 * cublas_ut_helper.cu
 *
 *  Created on: May 30, 2014
 *      Author: mike
 */

#include "inc/cublas_ut_helper.cuh"

#include <stdexcept>

using namespace std;

cublasHandle_t UTGetCublasHandle()
{
	static cublasHandle_t s_handle = 0;

	CUresult result = cuInit(0);

	if (!s_handle)
	{
		cublasStatus_t /*status = cublasInit();*/

		status = cublasCreate(&s_handle);

		if (status != CUBLAS_STATUS_SUCCESS)
			throw runtime_error("Unable to allocate the cublas handle");
	}

	return s_handle;
}
