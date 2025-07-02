#include "ThreadPool.h"
#include "System.h"

class ThreadLoop
{
    using ThreadInitFunction = typename ThreadPool::ThreadInitFunction;

    private:
    MPMCQueue<std::function<void()>>&   taskQueue;
    // For waiting
    std::atomic_uint32_t&               startedTaskCount;
    std::atomic_uint32_t&               runningTaskCount;
    std::condition_variable&            waitCondition;
    //
    ThreadInitFunction                  initFunction;
    uint32_t                            threadNumber;

    public:
    ThreadLoop(MPMCQueue<std::function<void()>>& taskQueue,
               std::atomic_uint32_t& startedTaskCount,
               std::atomic_uint32_t& runningTaskCount,
               std::condition_variable& waitCondition,
               ThreadInitFunction initFunction, uint32_t threadNumber);

    void operator()(std::stop_token token)
    {
        SystemThreadHandle handle = GetCurrentThreadHandle();
        std::thread::native_handle_type handleCPP = handle;

        if(initFunction) initFunction(handleCPP, threadNumber);
        startedTaskCount++;
        startedTaskCount.notify_all();

        while(!token.stop_requested() &&
              !taskQueue.IsTerminated())
        {
            std::function<void()> work;
            taskQueue.Dequeue(work);
            runningTaskCount++;
            if(work) work();
            runningTaskCount--;

            if(runningTaskCount == 0 && taskQueue.IsEmpty())
            {
                waitCondition.notify_all();
            }
        }
    }
};

ThreadLoop::ThreadLoop(MPMCQueue<std::function<void()>>& taskQueue,
                       std::atomic_uint32_t& startedTaskCount,
                       std::atomic_uint32_t& runningTaskCount,
                       std::condition_variable& waitCondition,
                       ThreadInitFunction initFunction,
                       uint32_t threadNumber)
    : taskQueue(taskQueue)
    , startedTaskCount(startedTaskCount)
    , runningTaskCount(runningTaskCount)
    , waitCondition(waitCondition)
    , initFunction(initFunction)
    , threadNumber(threadNumber)
{}

ThreadPool::ThreadPool(size_t queueSize)
    : poolAllocator(&baseAllocator)
    , taskQueue(queueSize)
    , startedTaskCount(0)
    , runningTaskCount(0)
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

    // TODO: This may be wrong, at least it feels like it.
    // Check it later
    std::unique_lock<std::mutex> lock(waitMutex);
    waitCondition.wait(lock, [&]() -> bool
    {
        return (runningTaskCount == 0) && taskQueue.IsEmpty();
    });
}

void ThreadPool::RestartThreadsImpl(uint32_t threadCount, ThreadInitFunction initFunc)
{
    Wait();
    taskQueue.Terminate();
    threads.clear();
    taskQueue.RemoveQueuedTasks(true);

    threads.reserve(threadCount);
    for(uint32_t i = 0; i < threadCount; i++)
    {
        threads.emplace_back(ThreadLoop(taskQueue, startedTaskCount,
                                        runningTaskCount, waitCondition,
                                        initFunc, i));
    }

    while(startedTaskCount < threadCount)
        startedTaskCount.wait(0);
}

void ThreadPool::ClearTasks()
{
    taskQueue.RemoveQueuedTasks(false);
}
