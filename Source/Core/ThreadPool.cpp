#include "ThreadPool.h"
#include "System.h"

class ThreadLoop
{
    using ThreadInitFunction = typename ThreadPool::ThreadInitFunction;

    private:
    MPMCQueue<std::function<void()>>&   taskQueue;
    ThreadInitFunction                  initFunction;
    uint32_t                            threadNumber;

    public:
    ThreadLoop(MPMCQueue<std::function<void()>>& taskQueue,
               ThreadInitFunction initFunction, uint32_t threadNumber);

    void operator()(std::stop_token token)
    {
        SystemThreadHandle handle = GetCurrentThreadHandle();
        std::thread::native_handle_type handleCPP = handle;

        if(initFunction) initFunction(handleCPP, threadNumber);

        while(!token.stop_requested() &&
              !taskQueue.IsTerminated())
        {
            std::function<void()> work;
            taskQueue.Dequeue(work);
            if(work) work();
        }
    }
};

ThreadLoop::ThreadLoop(MPMCQueue<std::function<void()>>& taskQueue,
                       ThreadInitFunction initFunction,
                       uint32_t threadNumber)
    : taskQueue(taskQueue)
    , initFunction(initFunction)
    , threadNumber(threadNumber)
{}

ThreadPool::ThreadPool(size_t queueSize)
    : poolAllocator(&baseAllocator)
    , taskQueue(queueSize)
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

    std::future<void> endToken = SubmitTask([](){});
    endToken.wait();
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
        threads.emplace_back(ThreadLoop(taskQueue, initFunc, i));
    }
}

void ThreadPool::ClearTasks()
{
    taskQueue.RemoveQueuedTasks(false);
}
