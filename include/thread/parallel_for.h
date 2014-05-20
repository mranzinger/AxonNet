/*
 * File description: parallel_for.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <atomic>
#include <exception>

#include "thread_pool.h"

template<typename NumType, typename Fn>
void ParallelFor(CThreadPool &a_pool, NumType a_lowerBdInclusive, NumType a_upperBdExclusive, NumType a_increment, Fn a_handler)
{
    std::exception_ptr l_unhandledException;

    std::atomic<NumType> l_procCt(0);

    for (NumType i = a_lowerBdInclusive; i < a_upperBdExclusive; i += a_increment)
    {
        a_pool.QueueTask([&, i]
        {
            try
            {
                // Invoke the handler with the current iteration
                a_handler(i);

                // Increment the counter
                l_procCt += a_increment;
            }
            catch (std::exception &)
            {
                l_unhandledException = std::current_exception();
            }
        });

        // TODO: Add task recycling because the only thing that changes is the 'i' capture between
        // executions, and we wouldn't have to allocate anything
    }

    // Now wait for the tasks to finish. See 'parallel_for_each.h' for explanation
    // of usage
    a_pool.ProcessWhile([&] { return l_procCt < a_upperBdExclusive; });

    if (nullptr != l_unhandledException)
        std::rethrow_exception(l_unhandledException);
}

template<typename NumType, typename Fn>
void ParallelFor(NumType a_lowerBdInclusive, NumType a_upperBdExclusive, NumType a_increment, Fn a_handler)
{
    CThreadPool l_pool;

    ParallelFor(l_pool, a_lowerBdInclusive, a_upperBdExclusive, a_increment, std::move(a_handler));
}

template<typename NumType, typename Fn>
void FastFor(CThreadPool &a_pool, NumType a_lowerBdInclusive, NumType a_upperBdExclusive, NumType a_increment, Fn a_handler)
{
#ifdef _DEBUG
    for (NumType i = a_lowerBdInclusive; i < a_upperBdExclusive; i += a_increment)
    {
        a_handler(i);
    }
#else
    ParallelFor(a_pool, a_lowerBdInclusive, a_upperBdExclusive, a_increment, a_handler);
#endif
}

template<typename NumType, typename Fn>
void FastFor(NumType a_lowerBdInclusive, NumType a_upperBdExclusive, NumType a_increment, Fn a_handler)
{
#ifdef _DEBUG
    for (NumType i = a_lowerBdInclusive; i < a_upperBdExclusive; i += a_increment)
    {
        a_handler(i);
    }
#else
    ParallelFor(a_lowerBdInclusive, a_upperBdExclusive, a_increment, a_handler);
#endif
}



