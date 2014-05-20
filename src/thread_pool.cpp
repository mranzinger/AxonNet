/*
 * File description: thread_pool.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "thread/thread_pool.h"

#include <boost/bind.hpp>

CThreadPool::CThreadPool(size_t a_numThreads /*= 0*/)
    : m_taskQueue(100), m_allowAdd(true)
{
    // If 0 is passed, then allocate the number of threads equal to the number of compute units
    if (a_numThreads == 0)
        a_numThreads = sysconf(_SC_NPROCESSORS_ONLN);

    for (size_t i = 0; i < a_numThreads; ++i)
    {
        m_threads.create_thread(boost::bind(&CThreadPool::p_Run, this));
    }
}

CThreadPool::~CThreadPool()
{
    Kill();
}

void CThreadPool::Kill()
{
    // Don't allow anything else to be added to the queue
    m_allowAdd = false;

    // Wake up all the threads. This is useful if the queue is empty
    m_processEvt.notify_all();

    m_threads.join_all();
}

void CThreadPool::p_Run()
{


    Invokable *l_currInvoke = nullptr;
    while (true)
    {
        bool l_gotItem = m_taskQueue.pop(l_currInvoke);

        if (!l_gotItem)
        {
            // If the queue was empty, and nothing can be added,
            // then this worker is done
            if (!m_allowAdd)
                return;

            boost::unique_lock<boost::mutex> l_lock(m_processMutex);
            m_processEvt.wait(l_lock);
            continue;
        }

        try
        {
            // Execute the task
            (*l_currInvoke)();

            delete l_currInvoke;
        }
        catch (...)
        {
            delete l_currInvoke;
            // TODO: cerr?
            throw;
        }
    }
}

void CThreadPool::ProcessOne()
{
    Invokable *l_currInvoke = nullptr;

    bool l_gotItem = m_taskQueue.pop(l_currInvoke);

    if (!l_gotItem)
        return;

    try
    {
        (*l_currInvoke)();

        delete l_currInvoke;
    }
    catch (...)
    {
        delete l_currInvoke;

        throw;
    }
}


size_t CThreadPool::NumThreads() const
{
    return m_threads.size();
}


