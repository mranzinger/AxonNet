/*
 * mat_preserver.cu
 *
 *  Created on: Jun 7, 2014
 *      Author: mike
 */

#include "mat_preserver.cuh"

MatPreserver& MatPreserver::Instance()
{
	static MatPreserver *s_instance = NULL;

	if (!s_instance)
		s_instance = new MatPreserver;

	return *s_instance;
}
