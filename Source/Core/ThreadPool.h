#pragma once

// Simple thread pool. Static sized allocation of threads
#include <thread>
#include <vector>
#include <functional>
#include <barrier>
#include <future>
#include <memory_resource>
#include <variant>

#include "MPMCQueue.h"
#include "Math.h"
#include "Error.h"

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
    void WaitAll() const;
    bool AnyValid() const;
};

class ThreadPool
{
    public:
    using ThreadInitFunction = std::function<void(std::thread::native_handle_type, uint32_t)>;
    using BlockWorkFunction = std::function<void(uint32_t, uint32_t)>;
    static constexpr uint32_t DefaultQueueSize = 512;

    private:
    std::vector<std::jthread>   threads;
    //
    std::pmr::monotonic_buffer_resource                 baseAllocator;
    std::pmr::unsynchronized_pool_resource              poolAllocator;
    //
    MPMCQueue<std::function<void()>>                    taskQueue;

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
                    ThreadPool(ThreadPool&&) = default;
    ThreadPool&     operator=(const ThreadPool&) = delete;
    ThreadPool&     operator=(ThreadPool&&) = default;
                    ~ThreadPool() = default;

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
    using ResultT = std::invoke_result_t<WorkFunc, uint32_t, uint32_t>;

    // Determine partition size etc..
    if(partitionCount == 0)
        partitionCount = static_cast<uint32_t>(threads.size());
    partitionCount = std::min(totalWorkSize, partitionCount);
    uint32_t sizePerPartition = Math::DivideUp(totalWorkSize, partitionCount);

    // Store the work functor somewhere safe.
    // Work functor will be copied once to a shared_ptr,
    // used multiple times, then gets deleted automatically
    auto sharedWork = std::allocate_shared<BlockWorkFunction>(std::pmr::polymorphic_allocator<BlockWorkFunction>(&poolAllocator),
                                                              std::forward<WorkFunc>(wf));

    // Enqueue the type ereased works to the queue
    MultiFuture<ResultT> result;
    result.futures.reserve(partitionCount);
    for(uint32_t i = 0; i < partitionCount; i++)
    {
        uint32_t start = i * sizePerPartition;
        uint32_t end = std::min((i + 1) * sizePerPartition, totalWorkSize);
        using AllocT = std::pmr::polymorphic_allocator<std::promise<ResultT>>;
        auto promise = std::allocate_shared<std::promise<ResultT>>(AllocT(&poolAllocator));

        std::future<ResultT> future = promise->get_future();
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
    return result;
}

template<ThreadTaskWorkC WorkFunc>
std::future<std::invoke_result_t<WorkFunc>>
ThreadPool::SubmitTask(WorkFunc&& wf)
{
    using ResultT = std::invoke_result_t<WorkFunc>;
    using AllocT = std::pmr::polymorphic_allocator<std::promise<ResultT>>;
    auto promise = std::allocate_shared<std::promise<ResultT>>(AllocT(&poolAllocator));
    std::future<ResultT> result = promise->get_future();

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
    return result;
}

template<ThreadDetachableTaskWorkC WorkFunc>
void ThreadPool::SubmitDetachedTask(WorkFunc&& wf)
{
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
