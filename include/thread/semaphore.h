/*
 * File description: semaphore.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

class CSemaphore
{
private:
    unsigned int m_initialCount, m_count;
    boost::mutex m_mutex;
    boost::condition_variable m_condition;

public:
    explicit CSemaphore(unsigned int initialCount)
        : m_initialCount(initialCount), m_count(initialCount)
    {
    }
    CSemaphore(unsigned int initialCount, unsigned int maxCount)
        : m_initialCount(initialCount), m_count(maxCount)
    {
    }

    void Signal()
    {
        boost::unique_lock<boost::mutex> l_lock(m_mutex);

        ++m_count;

        // Wake up any waiting threads.
        // Always do this, even if count wasn't 0 on entry.
        // Otherwise, we might not wake up enough waiting threads if we
        // get a number of signal calls in a row.
        m_condition.notify_one();
    }

    void Wait()
    {
        boost::unique_lock<boost::mutex> l_lock(m_mutex);
        while (0 == m_count)
        {
            m_condition.wait(l_lock);
        }
        --m_count;
    }

    void WaitAll()
    {
        for (unsigned int i = 0; i < m_initialCount; ++i)
        {
            Wait();
        }

        // Once we have exited the loop, this function technically owns all of the handles.
        // Now, re-populate
        assert(m_count == 0);

        m_count = m_initialCount;

        m_condition.notify_all();
    }
};



