/*
 * File description: parallel_for_each.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <atomic>
#include <exception>

#include "thread_pool.h"

template<typename IterType, typename Fn>
void ParallelForEach(CThreadPool &a_pool, IterType a_iter, const IterType &a_end, Fn a_handler)
{
    std::exception_ptr l_unhandledException;

    // Create a semaphore for the size of the range.
    // This when when all processing is finished, WaitAll will return.
    // Start the free count at 0 so that each task completion can increment the count
    // until we reach capacity
    std::atomic<size_t> l_procCt(0);
    size_t l_endCt = a_end - a_iter;

    for (; a_iter != a_end; ++a_iter)
    {
        const auto &l_val = *a_iter;

        a_pool.QueueTask([&, l_val]
        {
            try
            {
                a_handler(l_val);

                ++l_procCt;
            }
            catch (std::exception &)
            {
                l_unhandledException = std::current_exception();
            }
        });
    }

    // Process thread pool tasks until the expected count has been reached.
    // This is also useful if the current thread is actually part of the pool
    // because it will prevent a thread from getting stalled. Downside is that
    // this turns into a spin lock while waiting for the last task in the pool.
    a_pool.ProcessWhile([&] { return l_procCt < l_endCt; });

    // If an exception occurred, then rethrow it back on the main thread
    if (nullptr != l_unhandledException)
        std::rethrow_exception(l_unhandledException);
}


template<typename IterType, typename Fn>
void ParallelForEach(IterType a_iter, const IterType &a_end, Fn a_handler)
{
    CThreadPool l_pool;

    ParallelFor(l_pool, std::move(a_iter), a_end, std::move(a_handler));
}


