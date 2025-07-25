#pragma once

// Simple thread pool. Static sized allocation of threads
#include <thread>
#include <vector>
#include <functional>
#include <future>
#include <memory_resource>

#include "Log.h"
#include "MPMCQueue.h"
#include "System.h"

// TODO: Check if std has these predefined somewhere?
// (i.e. concept "callable_as")
template<class T>
concept ThreadBlockWorkC = requires(const T& t)
{
    { t(uint32_t(), uint32_t()) } -> std::same_as<void>;
};

template<class T>
concept ThreadTaskWorkC = requires(const T& t)
{
    { t() } -> std::same_as<std::invoke_result_t<T>>;
};

template<class T>
concept ThreadDetachableTaskWorkC = requires(const T& t)
{
    { t() } -> std::same_as<void>;
};


template<class T>
concept ThreadInitFuncC = requires(const T& t)
{
    { t(std::thread::native_handle_type(), uint32_t()) } -> std::same_as<void>;
};

template<class T>
struct MultiFuture
{
    std::vector<std::future<T>> futures;
    //
    void            WaitAll() const;
    bool            AnyValid() const;
    std::vector<T>  GetAll();
};

template<>
struct MultiFuture<void>
{
    std::vector<std::future<void>> futures;
    //
    void WaitAll() const
    {
        for(const std::future<void>& f : futures)
        {
            f.wait();
        }
    }
    //
    bool AnyValid() const
    {
        // This is technically valid for "void" futures.
        // We allow user to submit empty work block works
        // It will return empty multi future
        if(futures.empty()) return true;

        for(const std::future<void>& f : futures)
        {
            if(f.valid()) return true;
        }
        return false;
    }
    //
    void GetAll()
    {
        if(futures.empty()) return;
        for(std::future<void>& f : futures)
        {
            f.get();
        }
        return;
    }
};

class ThreadPool
{
    public:
    using ThreadInitFunction = std::function<void(std::thread::native_handle_type, uint32_t)>;
    using BlockWorkFunction = std::function<void(uint32_t, uint32_t)>;
    static constexpr uint32_t DefaultQueueSize = 512;

    private:
    std::pmr::monotonic_buffer_resource     baseAllocator;
    std::pmr::synchronized_pool_resource    poolAllocator;
    //
    MPMCQueue<std::function<void()>>        taskQueue;
    //
    std::vector<std::jthread>               threads;
    std::atomic_uint64_t                    issuedTaskCount;
    // completedCounter is mostly written by worker threads,
    // issuedCounter is written by producer thread(s).
    // Putting a gap here should eliminate data transfer between
    // threads
    alignas(MRayCPUCacheLineDestructive)
    std::atomic_uint64_t                    completedTaskCount;

    void RestartThreadsImpl(uint32_t threadCount, ThreadInitFunction);

    protected:
    public:
    // Constructors & Destructor
                    ThreadPool(size_t queueSize = DefaultQueueSize);
                    ThreadPool(uint32_t threadCount,
                               size_t queueSize = DefaultQueueSize);
    template<ThreadInitFuncC InitFunction>
                    ThreadPool(uint32_t threadCount, InitFunction&&,
                               size_t queueSize = DefaultQueueSize);
                    ThreadPool(const ThreadPool&) = delete;
                    ThreadPool(ThreadPool&&) = delete;
    ThreadPool&     operator=(const ThreadPool&) = delete;
    ThreadPool&     operator=(ThreadPool&&) = delete;
                    ~ThreadPool();

    template<ThreadInitFuncC InitFunction>
    void            RestartThreads(uint32_t threadCount, InitFunction&&);
    uint32_t        ThreadCount() const;
    void            Wait();
    void            ClearTasks();



    //
    template<ThreadBlockWorkC WorkFunc>
    MultiFuture<void>
    SubmitBlocks(uint32_t totalWorkSize, WorkFunc&&,
                 uint32_t partitionCount = 0);
    //
    template<ThreadTaskWorkC WorkFunc>
    std::future<std::invoke_result_t<WorkFunc>>
    SubmitTask(WorkFunc&&);

    template<ThreadDetachableTaskWorkC WorkFunc>
    void SubmitDetachedTask(WorkFunc&&);
};

template<class T>
void MultiFuture<T>::WaitAll() const
{
    for(const std::future<T>& f : futures)
    {
        f.wait();
    }
}

template<class T>
bool MultiFuture<T>::AnyValid() const
{
    for(const std::future<T>& f : futures)
    {
        if(f.valid()) return true;
    }
    return false;
}

template<class T>
std::vector<T> MultiFuture<T>::GetAll()
{
    std::vector<T> result;
    result.reserve(futures.size());
    for(std::future<T>& f : futures)
    {
        result.emplace_back(f.get());
    }
    return result;
}

template<ThreadInitFuncC InitFunction>
ThreadPool::ThreadPool(uint32_t threadCount, InitFunction&& initFunction,
                       size_t queueSize)
    : ThreadPool(queueSize)
{
    RestartThreads(threadCount, initFunction);
}

template<ThreadInitFuncC InitFunction>
void ThreadPool::RestartThreads(uint32_t threadCount, InitFunction&& initFunction)
{
    RestartThreadsImpl(threadCount, std::forward<InitFunction>(initFunction));
}

template<ThreadBlockWorkC WorkFunc>
MultiFuture<void>
ThreadPool::SubmitBlocks(uint32_t totalWorkSize, WorkFunc&& wf,
                         uint32_t partitionCount)
{
    if(totalWorkSize == 0) return MultiFuture<void>{};

    using ResultT = std::invoke_result_t<WorkFunc, uint32_t, uint32_t>;

    // Determine partition size etc..
    if(partitionCount == 0)
        partitionCount = static_cast<uint32_t>(threads.size());
    partitionCount = std::min(totalWorkSize, partitionCount);
    uint32_t sizePerPartition = totalWorkSize / partitionCount;
    uint32_t residual = totalWorkSize - sizePerPartition * partitionCount;

    // Store the work functor somewhere safe.
    // Work functor will be copied once to a shared_ptr,
    // used multiple times, then gets deleted automatically
    issuedTaskCount.fetch_add(partitionCount);
    auto sharedWork = std::allocate_shared<BlockWorkFunction>(std::pmr::polymorphic_allocator<BlockWorkFunction>(&poolAllocator),
                                                              std::forward<WorkFunc>(wf));

    // Enqueue the type erased works to the queue
    MultiFuture<ResultT> result;
    result.futures.reserve(partitionCount);
    uint32_t startOffset = 0;
    for(uint32_t i = 0; i < partitionCount; i++)
    {
        uint32_t start = startOffset;
        uint32_t end = start + sizePerPartition;
        if(residual > 0)
        {
            end++;
            residual--;
        }
        startOffset = end;

        using AllocT = std::pmr::polymorphic_allocator<std::promise<ResultT>>;
        auto promise = std::allocate_shared<std::promise<ResultT>>(AllocT(&poolAllocator));
        auto future = promise->get_future();
        taskQueue.Enqueue([=]()
        {
            try
            {
                if constexpr(std::is_same_v<ResultT, void>)
                {
                    (*sharedWork)(start, end);
                    promise->set_value();
                }
                else
                {
                    promise->set_value((*sharedWork)(start, end));
                }
            }
            catch(...)
            {
                promise->set_exception(std::current_exception());
            }
        });
        result.futures.push_back(std::move(future));
    }
    assert(residual == 0);
    assert(startOffset == totalWorkSize);

    return result;
}

template<ThreadTaskWorkC WorkFunc>
std::future<std::invoke_result_t<WorkFunc>>
ThreadPool::SubmitTask(WorkFunc&& wf)
{
    issuedTaskCount.fetch_add(1);

    using ResultT = std::invoke_result_t<WorkFunc>;
    using AllocT = std::pmr::polymorphic_allocator<std::promise<ResultT>>;
    auto promise = std::allocate_shared<std::promise<ResultT>>(AllocT(&poolAllocator));
    auto future = promise->get_future();

    // Here we do not need to copy the functor on a shared location
    // since each task will be executed by a single thread
    // Let the std::function handle it
    taskQueue.Enqueue([=, wf = std::forward<WorkFunc>(wf)]()
    {
        try
        {
            if constexpr(std::is_same_v<ResultT, void>)
            {
                wf();
                promise->set_value();
            }
            else
            {
                promise->set_value(wf());
            }
        }
        catch(...)
        {
            promise->set_exception(std::current_exception());
        }
    });
    return future;
}

template<ThreadDetachableTaskWorkC WorkFunc>
void ThreadPool::SubmitDetachedTask(WorkFunc&& wf)
{
    issuedTaskCount.fetch_add(1);
    // Here we do not need to copy the functor on a shared location
    // since each task will be executed by a single thread
    // Let the std::function handle it
    taskQueue.Enqueue([wf = std::forward<WorkFunc>(wf)]()
    {
        try
        {
            wf();
        }
        catch(...)
        {
            MRAY_ERROR_LOG("Exception caught on a detached task! Terminating process");
            std::exit(1);
        }
    });
}
