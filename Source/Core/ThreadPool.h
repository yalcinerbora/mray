#pragma once

// Simple thread pool. Static sized allocation of threads
#include <thread>
#include <vector>
#include <functional>
#include <future>
#include <memory_resource>
#include <memory>
#include <array>

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

// Type erase via shared_ptr<void>
// We try to reduce allocation count with this.
// TODO: Group promise/callable to a single shared pointer later
struct TPCallable
{
    using Deleter       = void(*)(void);
    //using InternalFunc  = std::function<void(const void*, void*, uint32_t, uint32_t)>;
    using InternalFunc  = void(*)(const void*, void*, uint32_t, uint32_t);
    //
    std::shared_ptr<void>       workFunc;
    std::shared_ptr<void>       promise;
    uint32_t                    start   = 0;
    uint32_t                    end     = 0;
    InternalFunc                call    = nullptr;

    template<ThreadBlockWorkC T, class R>
    static void BlockTaskCall(const void* workFuncRaw, void* promiseRaw,
                              uint32_t start, uint32_t end)
    {
        using P = std::promise<R>;
        const T* wfPtr = reinterpret_cast<const T*>(workFuncRaw);
        P* promisePtr = reinterpret_cast<P*>(promiseRaw);
        wfPtr = std::launder(wfPtr); promisePtr = std::launder(promisePtr);
        const T& wf = *wfPtr;
        P& promise = *promisePtr;
        try
        {
            if constexpr(std::is_same_v<R, void>)
            {
                wf(start, end);
                promise.set_value();
            }
            else
            {
                promise.set_value(wf(start, end));
            }
        }
        catch(...)
        {
            promise.set_exception(std::current_exception());
        }
    }

    template<ThreadTaskWorkC T, class R>
    static void TaskCall(const void* workFuncRaw, void* promiseRaw, uint32_t, uint32_t)
    {
        using P = std::promise<R>;
        const T* wfPtr = reinterpret_cast<const T*>(workFuncRaw);
        P* promisePtr = reinterpret_cast<P*>(promiseRaw);
        wfPtr = std::launder(wfPtr); promisePtr = std::launder(promisePtr);
        const T& wf = *wfPtr;
        P& promise = *promisePtr;
        try
        {
            if constexpr(std::is_same_v<R, void>)
            {
                wf();
                promise.set_value();
            }
            else
            {
                promise.set_value(wf());
            }
        }
        catch(...)
        {
            promise.set_exception(std::current_exception());
        }
    }

    template<ThreadDetachableTaskWorkC T>
    static void DetachedTaskCall(const void* workFuncRaw, void*, uint32_t, uint32_t)
    {
        const T* wfPtr = reinterpret_cast<const T*>(workFuncRaw);
        wfPtr = std::launder(wfPtr);
        const T& wf = *wfPtr;
        try
        {
            wf();
        }
        catch(...)
        {
            MRAY_ERROR_LOG("Exception caught on a detached task! Terminating process");
            std::exit(1);
        }
    }

    public:
    // Constructors & Destructor
    // TODO: How to define this outside of the class?
    // Type erasure via templated constructor
    // Destruction is handled by shared_ptr automatically.
    template<ThreadBlockWorkC T, class R>
    TPCallable(std::shared_ptr<T> work, std::shared_ptr<std::promise<R>> p,
               uint32_t start, uint32_t end)
        : workFunc(work)
        , promise(p)
        , start(start)
        , end(end)
    {
        assert(work != nullptr);
        assert(p != nullptr);
        call = static_cast<InternalFunc>(&BlockTaskCall<T, R>);
    }
    template<ThreadTaskWorkC T, class R>
    TPCallable(std::shared_ptr<T> work, std::shared_ptr<std::promise<R>> p)
        : workFunc(work)
        , promise(p)
        , start(0)
        , end(0)
    {
        assert(work != nullptr);
        assert(p != nullptr);
        call = static_cast<InternalFunc>(&TPCallable::TaskCall<T, R>);
    }

    template<ThreadDetachableTaskWorkC T>
    TPCallable(std::shared_ptr<T> work)
        : workFunc(work)
        , promise(nullptr)
        , start(0)
        , end(0)
    {
        assert(work != nullptr);
        call = static_cast<InternalFunc>(&TPCallable::DetachedTaskCall<T>);
    }
    // Constructors & Destructor continued...
                TPCallable() = default;
                TPCallable(const TPCallable&) = delete;
                TPCallable(TPCallable&&) = default;
    TPCallable& operator=(const TPCallable&) = delete;
    TPCallable& operator=(TPCallable&&) = default;
                ~TPCallable() = default;

    void operator()() const;
};

class ThreadPool
{
    public:
    using ThreadInitFunction = std::function<void(std::thread::native_handle_type, uint32_t)>;
    static constexpr uint32_t DefaultQueueSize = 512;

    private:
    std::pmr::monotonic_buffer_resource     baseAllocator;
    std::pmr::synchronized_pool_resource    poolAllocator;
    //
    MPMCQueue<TPCallable>                   taskQueue;
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

inline
void TPCallable::operator()() const
{
    if(call) call(workFunc.get(), promise.get(), start, end);
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

    // Determine partition size etc..
    if(partitionCount == 0)
        partitionCount = static_cast<uint32_t>(threads.size());
    partitionCount = std::min(totalWorkSize, partitionCount);
    uint32_t sizePerPartition = totalWorkSize / partitionCount;
    uint32_t residual = totalWorkSize - sizePerPartition * partitionCount;
    issuedTaskCount.fetch_add(partitionCount);

    // Store the work functor somewhere safe.
    // Work functor will be copied once to a shared_ptr,
    // used multiple times, then gets deleted automatically
    // TODO: Clang has a bug in which it does not generate
    // code for some functions.
    // https://github.com/llvm/llvm-project/issues/57561
    // Wrapping it to std function fixes the issue
    // (Although defeats the entire purpose of the new system...)
    //using WFType = std::remove_cvref_t<WorkFunc>;
    using WFType = std::function<void(uint32_t, uint32_t)>;
    auto sharedWork = std::allocate_shared<WFType>(std::pmr::polymorphic_allocator<WFType>(&poolAllocator),
                                                   std::forward<WorkFunc>(wf));

    // Enqueue the type erased works to the queue
    MultiFuture<void> result;
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

        using AllocT = std::pmr::polymorphic_allocator<std::promise<void>>;
        auto promise = std::allocate_shared<std::promise<void>>(AllocT(&poolAllocator));
        auto future = promise->get_future();

        taskQueue.Enqueue(TPCallable(sharedWork, promise, start, end));
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

    using WFType = std::remove_cvref_t<WorkFunc>;
    using ResultT = std::invoke_result_t<WFType>;
    using AllocT = std::pmr::polymorphic_allocator<std::promise<ResultT>>;
    // Store the work functor somewhere safe.
    // Work functor will be copied once to a shared_ptr,
    // used multiple times, then gets deleted automatically
    auto sharedWork = std::allocate_shared<WFType>(std::pmr::polymorphic_allocator<WFType>(&poolAllocator),
                                                   std::forward<WorkFunc>(wf));
    auto promise = std::allocate_shared<std::promise<ResultT>>(AllocT(&poolAllocator));
    auto future = promise->get_future();

    taskQueue.Enqueue(TPCallable(sharedWork, promise));
    return future;
}

template<ThreadDetachableTaskWorkC WorkFunc>
void ThreadPool::SubmitDetachedTask(WorkFunc&& wf)
{
    issuedTaskCount.fetch_add(1);
    using WFType = std::remove_cvref_t<WorkFunc>;
    // Store the work functor somewhere safe.
    // Work functor will be copied once to a shared_ptr,
    // used multiple times, then gets deleted automatically
    auto sharedWork = std::allocate_shared<WFType>(std::pmr::polymorphic_allocator<WFType>(&poolAllocator),
                                                   std::forward<WorkFunc>(wf));
    taskQueue.Enqueue(TPCallable(sharedWork));
}
