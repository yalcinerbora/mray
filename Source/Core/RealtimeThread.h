#pragma once
/**

RealtimeThread Thread Partial Interface

used by threads that do same work over and over again
(currently Tracer only, visor is GUI app so MT is better for it
due to dear imgui's MT (more or less) requirement)

Extender can define internal terminate condition
where thread automatically ends

TODO:
This was before jthread, (this has extra this has pause functionality)
Probably jthread can be used instead. (We do not pause via this interface,
instead we pause using the queue's condition variable)
*/

#include <thread>
#include <mutex>
#include <condition_variable>

#include "System.h"

class RealtimeThread
{
    private:
    std::thread                 thread;
    std::mutex                  mutex;
    std::condition_variable     conditionVar;
    bool                        stopSignal;
    bool                        pauseSignal;

    void                        THRDEntry();

    protected:
    virtual bool                InternallyTerminated() const = 0;
    virtual void                InitialWork() = 0;
    virtual void                LoopWork() = 0;
    virtual void                FinalWork() = 0;

    public:
    // Constructors & Destructor
                                RealtimeThread();
    virtual                     ~RealtimeThread();

    void                        Start(const std::string& name);
    void                        Stop();
    void                        Pause(bool pause);

    bool                        IsTerminated();
};

inline void RealtimeThread::THRDEntry()
{
    InitialWork();
    while(!InternallyTerminated() && !stopSignal)
    {
        LoopWork();

        // Condition
        {
            std::unique_lock<std::mutex> lock(mutex);
            // Wait if paused (or stop signal is set)
            conditionVar.wait(lock, [&]
            {
                return stopSignal || !pauseSignal;
            });
        }
    }
    FinalWork();
}

inline RealtimeThread::RealtimeThread()
    : stopSignal(false)
    , pauseSignal(false)
{}

inline RealtimeThread::~RealtimeThread()
{
    Stop();
}

inline void RealtimeThread::Start(const std::string& name)
{
    // If the system was previously paused
    // ignore that
    pauseSignal = false;

    // Launch a new thread
    thread = std::thread(&RealtimeThread::THRDEntry, this);
    if(!name.empty())
        RenameThread(thread.native_handle(), name);
}

inline void RealtimeThread::Stop()
{
    mutex.lock();
    stopSignal = true;
    mutex.unlock();
    conditionVar.notify_one();
    if(thread.joinable()) thread.join();
    stopSignal = false;
}

inline void RealtimeThread::Pause(bool pause)
{
    mutex.lock();
    pauseSignal = pause;
    mutex.unlock();
    conditionVar.notify_one();
}

inline bool RealtimeThread::IsTerminated()
{
    return InternallyTerminated();
}