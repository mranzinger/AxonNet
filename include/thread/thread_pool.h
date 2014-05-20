/*
 * File description: thread_pool.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#pragma once

#include <vector>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/thread/condition_variable.hpp>

class CThreadPool
{
    // Inner Class Definitions
private:
    class Invokable
    {
    public:
        virtual ~Invokable() { }

        virtual void operator()() = 0;
    };

    template<typename Fn>
    class FnInvokable : public Invokable
    {
    private:
        Fn m_fn;

    public:
        FnInvokable(Fn fn)
            : m_fn(std::move(fn)) { }

        virtual void operator()()
        {
            m_fn();
        }
    };

private:
    boost::thread_group m_threads;
    boost::lockfree::queue<Invokable*, boost::lockfree::fixed_sized<false>> m_taskQueue;
    boost::condition_variable m_processEvt;
    boost::mutex m_processMutex;
    bool m_allowAdd;

public:
    explicit CThreadPool(size_t numThreads = 0);
    ~CThreadPool();

    // Prevent copying
    CThreadPool(const CThreadPool &) = delete;
    CThreadPool &operator=(const CThreadPool &) = delete;

    size_t NumThreads() const;

    void Kill();

    // Processes a task on the calling thread. This can be used
    // while waiting for some other operation to be completed without
    // suspending this thread
    void ProcessOne();

    // Processes tasks on the calling thread until the predicate returns false.
    // Signature of predicate must be 'bool ()'
    template<typename Predicate>
    void ProcessWhile(Predicate a_predicate)
    {
        do
        {
            ProcessOne();
        }
        while (a_predicate());
    }

    // Adds a task that will be run on a pooled thread. The function cannot take any
    // arguments
    template<typename Fn>
    bool QueueTask(Fn &&a_fn)
    {
        if (!m_allowAdd)
            return false;

        Invokable *l_pItem = nullptr;

        try
        {
            l_pItem = new FnInvokable<Fn>(std::forward<Fn>(a_fn));

            m_taskQueue.push(l_pItem);

            // Wake up a sleeping thread to process the event
            m_processEvt.notify_one();
        }
        catch (...)
        {
            delete l_pItem;
            throw;
        }

        return true;
    }

private:


    void p_Run();
};


