#include "ThreadPool.h"
#include "System.h"

inline void AtomicWaitValueToAchieve(std::atomic_uint64_t& checkLoc, uint64_t valToAchieve)
{
    // For explanation check GPUSystemCPU.h,
    // "AtomicWaitValueToAchieve" function.
    for(uint64_t check = checkLoc; check < valToAchieve; check = checkLoc)
    {
        checkLoc.wait(check);
    }
}

class ThreadLoop
{
    using ThreadInitFunction = typename ThreadPool::ThreadInitFunction;

    private:
    MPMCQueueAtomic<TPCallable>&  taskQueue;
    // For waiting
    std::atomic_uint64_t&   completedTaskCount;
    //
    ThreadInitFunction      initFunction;
    uint32_t                threadNumber;

    public:
    ThreadLoop(MPMCQueueAtomic<TPCallable>& taskQueue,
               std::atomic_uint64_t& completedTaskCount,
               ThreadInitFunction initFunction, uint32_t threadNumber);

    void operator()(std::stop_token token)
    {
        SystemThreadHandle handle = GetCurrentThreadHandle();
        std::thread::native_handle_type handleCPP = handle;

        if(initFunction) initFunction(handleCPP, threadNumber);

        // Signal that you are in your event loop
        completedTaskCount.fetch_add(1);
        completedTaskCount.notify_all();

        while(!token.stop_requested() && !taskQueue.IsTerminated())
        {
            TPCallable work;
            taskQueue.Dequeue(work);
            work();
            // We do not know if anyone waiting for the
            // the value here so we directly notify
            completedTaskCount.fetch_add(1);
            completedTaskCount.notify_all();
        }
    }
};

ThreadLoop::ThreadLoop(MPMCQueueAtomic<TPCallable>& taskQueue,
                       std::atomic_uint64_t& completedTaskCount,
                       ThreadInitFunction initFunction,
                       uint32_t threadNumber)
    : taskQueue(taskQueue)
    , completedTaskCount(completedTaskCount)
    , initFunction(initFunction)
    , threadNumber(threadNumber)
{}

ThreadPool::ThreadPool(size_t queueSize)
    : poolAllocator(&baseAllocator)
    , taskQueue(queueSize)
    , issuedTaskCount(0)
    , completedTaskCount(0)
{}

ThreadPool::ThreadPool(uint32_t threadCount, size_t queueSize)
    : ThreadPool(queueSize)
{
    RestartThreads(threadCount, ThreadInitFunction());
}

ThreadPool::~ThreadPool()
{
    // Wait all jobs to finish
    Wait();
    // Terminate the task queue,
    // Threads may be waiting over the condition variable
    taskQueue.Terminate();
    threads.clear();
}

uint32_t ThreadPool::ThreadCount() const
{
    return static_cast<uint32_t>(threads.size());
}

void ThreadPool::Wait()
{
    if(threads.empty()) return;

    uint64_t curIssueCount = issuedTaskCount.load();
    AtomicWaitValueToAchieve(completedTaskCount, curIssueCount);
}

void ThreadPool::RestartThreadsImpl(uint32_t threadCount, ThreadInitFunction initFunc)
{
    Wait();
    taskQueue.Terminate();
    threads.clear();
    taskQueue.RemoveQueuedTasks(true);

    completedTaskCount.store(0);
    threads.reserve(threadCount);
    for(uint32_t i = 0; i < threadCount; i++)
    {
        threads.emplace_back(ThreadLoop(taskQueue, completedTaskCount,
                                        initFunc, i));
    }
    // Issue phony init tasks, thread loop will increment these
    issuedTaskCount.store(threadCount);
    AtomicWaitValueToAchieve(completedTaskCount, threadCount);
}

void ThreadPool::ClearTasks()
{
    taskQueue.RemoveQueuedTasks(false);
}
