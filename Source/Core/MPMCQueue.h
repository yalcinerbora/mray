#pragma once

/**

Fix Sized Multi-producer Multi-consumer ring buffer queue,
simple implementation (single lock and cond var used)

*/

#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>

template<class T>
class MPMCQueue
{
    private:
        std::vector<T>          data;

        size_t                  enqueueLoc;
        size_t                  dequeueLoc;

        std::condition_variable_any enqueueWake;
        std::condition_variable_any dequeueWake;

        std::timed_mutex        mutex;
        std::atomic_bool        isTerminated;

        bool                    IsEmptyUnsafe();
        bool                    IsFullUnsafe();
        void                    Increment(size_t&);

    protected:
    public:
        // Constructors & Destructor
                                MPMCQueue(size_t bufferSize);
                                MPMCQueue(const MPMCQueue&) = delete;
                                MPMCQueue(MPMCQueue&&) = delete;
        MPMCQueue&              operator=(const MPMCQueue&) = delete;
        MPMCQueue&              operator=(MPMCQueue&&) = delete;
                                ~MPMCQueue() = default;

        // Interface
        void                    Dequeue(T&);
        bool                    TryDequeue(T&);
        template<class D>
        bool                    TryDequeue(T&, D);
        void                    Enqueue(T&&);
        bool                    TryEnqueue(T&&);
        //
        bool                    IsEmpty();
        bool                    IsFull();

        // Awakes all threads and forces them to leave queue
        void                    Terminate();
        bool                    IsTerminated() const;
        // Removes the queued tasks
        void                    RemoveQueuedTasks(bool reEnable);
};

template<class T>
bool MPMCQueue<T>::IsEmptyUnsafe()
{
    return ((dequeueLoc + 1) % data.size())  == enqueueLoc;
}

template<class T>
bool MPMCQueue<T>::IsFullUnsafe()
{
    return enqueueLoc == dequeueLoc;
}

template<class T>
void MPMCQueue<T>::Increment(size_t& i)
{
    i = (i + 1) % data.size();
}

template<class T>
MPMCQueue<T>::MPMCQueue(size_t bufferSize)
    : data(bufferSize + 1)
    , enqueueLoc(1)
    , dequeueLoc(0)
    , isTerminated(false)
{
    assert(bufferSize != 0);
}

template<class T>
void MPMCQueue<T>::Dequeue(T& item)
{
    if (isTerminated) return;
    {
        std::unique_lock<std::timed_mutex> lock(mutex);
        dequeueWake.wait(lock, [&]()
        {
            return (!IsEmptyUnsafe() || isTerminated);
        });
        if (isTerminated) return;

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryDequeue(T& item)
{
    if (isTerminated) return false;
    {
        std::unique_lock<std::timed_mutex> lock(mutex);
        if(IsEmptyUnsafe() || isTerminated) return false;

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueueWake.notify_one();
    return true;
}

template<class T>
template<class D>
bool MPMCQueue<T>::TryDequeue(T& item, D duration)
{
    if(isTerminated) return false;
    {
        std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
        bool result = lock.try_lock_for(duration);
        //
        if(IsEmptyUnsafe() || isTerminated || !result)
            return false;

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueueWake.notify_one();
    return true;
}

template<class T>
void MPMCQueue<T>::Enqueue(T&& item)
{
    {
        std::unique_lock<std::timed_mutex> lock(mutex);
        enqueueWake.wait(lock, [&]()
        {
            return (!IsFullUnsafe() || isTerminated);
        });
        if (isTerminated) return;

        data[enqueueLoc] = std::move(item);
        Increment(enqueueLoc);
    }
    dequeueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryEnqueue(T&& item)
{
    if (isTerminated) return false;
    {
        std::unique_lock<std::timed_mutex> lock(mutex);
        if(IsFullUnsafe() || isTerminated) return false;

        data[enqueueLoc] = std::move(item);
        Increment(enqueueLoc);
    }
    dequeueWake.notify_one();
    return true;
}

template<class T>
void MPMCQueue<T>::Terminate()
{
    // TODO: There was a rare race condition
    // due to isTerminated is not guarded with
    // mutex? (It is an atomic variable)
    // Some threads do not see the is terminated value
    // although assignment operator should be "seq_cst"?
    // Anyway, this may be a problem in the future
    {
        std::unique_lock<std::timed_mutex> lock(mutex);
        isTerminated = true;
    }
    dequeueWake.notify_all();
    enqueueWake.notify_all();
}

template<class T>
bool MPMCQueue<T>::IsTerminated() const
{
    return isTerminated;
}

template<class T>
bool MPMCQueue<T>::IsEmpty()
{
    std::unique_lock<std::timed_mutex> lock(mutex);
    return IsEmptyUnsafe();
}

template<class T>
bool MPMCQueue<T>::IsFull()
{
    std::unique_lock<std::timed_mutex> lock(mutex);
    return IsFullUnsafe();
}

template<class T>
void MPMCQueue<T>::RemoveQueuedTasks(bool reEnable)
{
    std::unique_lock<std::timed_mutex> lock(mutex);
    for(T& t : data)
    {
        t = T();
    }
    enqueueLoc = 1;
    dequeueLoc = 0;

    if(reEnable) isTerminated = false;
}