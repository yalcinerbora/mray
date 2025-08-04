#pragma once

// Fix Sized Atomic Multi-producer Multi-consumer ring buffer queue.
// Somewhat advanced implementation. Given producer/consumer count
// vs. queue size ratio, atomic wait amount changes.
//
// Theoretically with infinite queue size this queue will be
// wait free (given producers/consumers keepup with each other).
// One problem is that this queue is wait only. You cannot give up
// producing/consuming after you atomically increment the queue.
//
// Similar queues exists in github. Similar to those designs,
//  this queue uses per-element atomic variable plus
//  atomic head and tail.
//
//  One different of this queue that we always increment the
//  head/tail (producer/consumer). Thus a producer will
//  reserve a slot on the time frame of the queue as such.
//
//  |---|---|---|---|
//    ^   ^   ^   ^
//   p0  p1  p2   p3
//   p4  p5  p6  ...
// Same is true for consumers as well. p0 will wait c0 to finish writing.
// (in initial case system behave as if c0 is already done).
//
// All in all we split the producers/consumers on to multiple single element
// queue's with a naive (round-robin) load balancing scheme.
//
// Problem with this approach is that when a producer/consumer commits
// to an enqueue/dequeue it has to finish it or the design collapses.
// So only other way to wake the waiting producer/consumer is to
// totally terminate the queue.
//
// Also T must be default constructible.
//
#include <cstdint>
#include <utility>
#include <vector>
#include <atomic>
#include <cassert>

#include "BitFunctions.h"
#include "System.h"

template<class T>
class MPMCQueueAtomic
{
    using a_uint64_t    = std::atomic_uint64_t;
    using a_bool_t      = std::atomic_bool;
    static constexpr uint64_t CONSUMED = 0;
    static constexpr uint64_t PRODUCED = 1;
    //
    struct alignas(MRayCPUCacheLineDestructive) QueueSlot
    {
        a_uint64_t  generation = 0;
        T           item;
    };

    private:
        std::vector<QueueSlot>  data;
        alignas(MRayCPUCacheLineDestructive)
        a_uint64_t              enqueueLoc;
        alignas(MRayCPUCacheLineDestructive)
        a_uint64_t              dequeueLoc;
        a_bool_t                isTerminated;

    protected:
    public:
        // Constructors & Destructor
                            MPMCQueueAtomic(size_t bufferSize);
                            MPMCQueueAtomic(const MPMCQueueAtomic&) = delete;
                            MPMCQueueAtomic(MPMCQueueAtomic&&) = delete;
        MPMCQueueAtomic&    operator=(const MPMCQueueAtomic&) = delete;
        MPMCQueueAtomic&    operator=(MPMCQueueAtomic&&) = delete;
                            ~MPMCQueueAtomic() = default;

        // Interface
        void Dequeue(T&);
        void Enqueue(T&&);
        //
        bool IsEmpty();
        bool IsFull();
        // Awakes all threads and forces them to leave queue
        void Terminate();
        bool IsTerminated() const;
        // Removes the queued tasks
        void RemoveQueuedTasks(bool reEnable);
};

template<class T>
MPMCQueueAtomic<T>::MPMCQueueAtomic(size_t bufferSize)
    : data(bufferSize)
    , enqueueLoc(0)
    , dequeueLoc(0)
    , isTerminated(false)
{
    // There has to be a one slot in the queue
    // to bit casting to work
    assert(bufferSize > 0);
}

template<class T>
void MPMCQueueAtomic<T>::Dequeue(T& item)
{
    if (isTerminated) return;
    // We do not have "try" functionality by design
    // Here we only atomic increment and guarantee a slot
    // Here we have minimal contention (no compswap loops etc.)
    uint64_t myLoc = dequeueLoc.fetch_add(1u);
    // Fallback of this is that we got the slot so we have to commit.
    // This means we must wait somewhere.
    uint64_t curGeneration = myLoc / data.size();
    uint64_t curLoc = myLoc % data.size();
    uint64_t genDesiredState = Bit::Compose<63, 1>(curGeneration, PRODUCED);

    // We will wait on the slot itself.
    // Slot has generation which can only be incremented
    // by a consumer. generation = N means
    // all the processing of all generation before N
    // (excluding N) is done.
    //QueueSlot& curElem = data[curLoc];
    QueueSlot& curElem = data.at(curLoc);
    a_uint64_t& atomicGen = curElem.generation;
    uint64_t expectedGen;
    //
    while((expectedGen = atomicGen.load()) != genDesiredState && !isTerminated)
        atomicGen.wait(expectedGen);
    //
    if(isTerminated) return;
    //
    // We got the required state, we can push the data now
    item = std::move(curElem.item);
    // Signal
    uint64_t genSignalState = Bit::Compose<63, 1>(curGeneration + 1, CONSUMED);
    atomicGen.store(genSignalState);
    atomicGen.notify_all();
}

template<class T>
void MPMCQueueAtomic<T>::Enqueue(T&& item)
{
    if (isTerminated) return;
    // We do not have "try" functionality by design
    // Here we only atomic increment and guarantee a slot
    // Here we have minimal contention (no compswap loops etc.)
    uint64_t myLoc = enqueueLoc.fetch_add(1u);
    // Fallback of this is that we got the slot so we have to commit.
    // This means we must wait somewhere.
    uint64_t curGeneration = myLoc / data.size();
    uint64_t curLoc = myLoc % data.size();
    uint64_t genDesiredState = Bit::Compose<63, 1>(curGeneration, CONSUMED);

    // We will wait on the slot itself.
    // Slot has generation which can only be incremented
    // by a consumer. generation = N means
    // all the processing of all generation before N
    // (excluding N) is done.
    //QueueSlot& curElem = data[curLoc];
    QueueSlot& curElem = data.at(curLoc);
    a_uint64_t& atomicGen = curElem.generation;
    uint64_t expectedGen;
    //
    while((expectedGen = atomicGen.load()) != genDesiredState && !isTerminated)
    //
        atomicGen.wait(expectedGen);
    //
    if(isTerminated) return;
    //
    // We got the required state, we can push the data now
    curElem.item = std::move(item);
    // Signal
    uint64_t genSignalState = Bit::Compose<63, 1>(curGeneration, PRODUCED);
    atomicGen.store(genSignalState);
    atomicGen.notify_all();
}

template<class T>
void MPMCQueueAtomic<T>::Terminate()
{
    isTerminated = true;
    for(QueueSlot& element : data)
    {
        // Guarantee a change on generation item
        // so that the notify wakes all threads
        element.generation++;
        element.generation.notify_all();
    }
}

template<class T>
bool MPMCQueueAtomic<T>::IsTerminated() const
{
    return isTerminated;
}

template<class T>
bool MPMCQueueAtomic<T>::IsEmpty()
{
    return (enqueueLoc - dequeueLoc) == 0;
}

template<class T>
bool MPMCQueueAtomic<T>::IsFull()
{
    return (enqueueLoc - dequeueLoc) == data.size();
}

template<class T>
void MPMCQueueAtomic<T>::RemoveQueuedTasks(bool reEnable)
{
    // TODO: Although this function works
    // it is not complete. We need to change the design
    // of the thread pool to remove the ABA problem.
    // also some form of cleanup is needed.

    // ***ABA Problem!***
    // Assume generation was 0
    // we increment it, so all the threads waiting over the generation
    // is awake (generation could've been any value so we increment)
    //isTerminated = true;
    for(QueueSlot& element : data)
    {
        element.generation = 0;
        element.item = T();
        element.generation.notify_all();
    }
    enqueueLoc = 0;
    dequeueLoc = 0;

    // ***ABA Problem! Cont.***
    // If someone was waiting for 0 and
    // couldn't wake up on time, will see the generation as zero
    // and sleep?
    // is awake (generation could've been any value so we increment)
    // for(QueueSlot& element : data)
    // {
    //     element.generation.notify_all();
    //     element.generation = 0;
    // }

    if(reEnable) isTerminated = false;
}