#pragma once

#include "GPUSystemCPU.h"
#include "../GPUSystem.h"

#include "Core/ThreadPool.h"

static constexpr uint32_t WarpSize()
{
    return 1u;
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MRAY_GPU MRAY_GPU_INLINE
static void WarpSynchronize() {}

MRAY_GPU MRAY_GPU_INLINE
static void BlockSynchronize() {}

MRAY_GPU MRAY_GPU_INLINE
static void ThreadFenceGrid()
{
    // TODO: Reason about this
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

namespace mray::host
{

MRAY_GPU MRAY_CGPU_INLINE
KernelCallParamsCPU::KernelCallParamsCPU()
    : gridSize(globalKCParams.gridSize)
    , blockSize(globalKCParams.blockSize)
    , blockId(globalKCParams.blockId)
    , threadId(globalKCParams.threadId)
{}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCPU::IssueKernelInternal(std::string_view name,
                                      uint32_t totalWorkCount,
                                      uint32_t tpb,
                                      //
                                      Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationCPU(domain, name);
    const auto annotationHandle = annotation.AnnotateScope();

    // We are locking here, since all of the block submissions
    // must happen in order.
    //
    // TODO: This is thread pools responsibility (We need to implement
    // ordered issue system, currently ThreadPool may issue
    // out-of-order blocks if there is another thread).
    //
    // This would require to change the MPMC queue so
    // skipping for now
    std::lock_guard _(cb->issueMutex);

    // Give just enough grid to eliminiate grid-stride / block-stride
    // loops (for CPU these do more harm than good, unlike GPU)

    uint32_t blockCount = DetermineGridStrideBlock(nullptr, 0, tpb,
                                                   totalWorkCount);
    uint32_t callerBlockCount = Math::DivideUp(totalWorkCount, tpb);
    //
    auto atomicKCounter = std::atomic_ref<uint64_t>(cb->issuedKernelCounter);
    uint64_t oldIssueCount = atomicKCounter.fetch_add(blockCount);

    // Issue kernel
    // "SubmitBlocks" does a "for loop Enqueue" calls which is thread-safe
    // but for this code to work, "for loop Enqueue" portion must be a
    // critical section, instead of the internal queue's data.
    // (see the above "TODO" section)
    auto* cbRef = &this->cb;
    tp->SubmitBlocks
    (
        blockCount,
        [
            ... fArgs = std::forward<Args>(fArgs),
            blockCount, oldIssueCount, cbRef,
            callerBlockCount, tpb
            //,name
        ](uint32_t blockStart, [[maybe_unused]] uint32_t blockEnd)
        {
            // Wait the previous kernel to finish
            // CUDA style queue emulation.
            auto atomicCompletedCounter = std::atomic_ref<uint64_t>((*cbRef)->completedKernelCounter);
            while(atomicCompletedCounter < oldIssueCount)
                atomicCompletedCounter.wait(std::numeric_limits<uint64_t>::max());

            std::atomic_thread_fence(std::memory_order_seq_cst);
            // I dont understand this wait, wait until "curIssueCount == oldIssueCount"
            // but what if we skip a beat and curIssueCount becomes larger than wait value?
            // Further thinking, how this is useful at all?
            // Given a value set I, A is technically all elements of the set but one
            // which is B.
            // "wait_predicate" would be better? Probably hard to implement, i dunno.
            // Anyway if this comment stays it works robustly.
            // TODO: Even if this there is no ABA issue, check the perf
            // and fall back to condition variable / mutex
            // compare as such "wait until (curIssueCount >= oldIssueCount)
            //
            // From this point on it previous kernels should be completed
            //
            //
            assert(blockEnd - blockStart == 1);
            globalKCParams.gridSize = callerBlockCount;
            globalKCParams.blockSize = tpb;

            // Do the block stride loop here,
            // each thread may be responsible for one or more blocks
            for(uint32_t bId = blockStart; bId < callerBlockCount; bId += blockCount)
            {
                globalKCParams.blockId = bId;
                for(uint32_t j = 0; j < tpb; j++)
                {
                    // For kernel call params to work
                    globalKCParams.threadId = j;
                    Kernel(fArgs...);
                }
            }
            atomicCompletedCounter.fetch_add(1);
            atomicCompletedCounter.notify_all();

            std::atomic_thread_fence(std::memory_order_seq_cst);
        },
        blockCount
    );
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCPU::IssueLambdaInternal(std::string_view name,
                                      uint32_t totalWorkCount,
                                      uint32_t tpb,
                                      Lambda&& func) const
{
    static const auto annotation = GPUAnnotationCPU(domain, name);
    const auto annotationHandle = annotation.AnnotateScope();

    // We are locking here, since all of the block submissions
    // must happen in order.
    //
    // TODO: This is thread pools responsibility (We need to implement
    // ordered issue system, currently ThreadPool may issue
    // out-of-order blocks if there is another thread).
    //
    // This would require to change the MPMC queue so
    // skipping for now
    std::lock_guard _(cb->issueMutex);

    // Give just enough grid to eliminiate grid-stride / block-stride
    // loops (for CPU these do more harm than good, unlike GPU)

    uint32_t blockCount = DetermineGridStrideBlock(nullptr, 0, tpb,
                                                   totalWorkCount);
    uint32_t callerBlockCount = Math::DivideUp(totalWorkCount, tpb);
    //
    auto atomicKCounter = std::atomic_ref<uint64_t>(cb->issuedKernelCounter);
    uint64_t oldIssueCount = atomicKCounter.fetch_add(blockCount);

    // Issue kernel
    // "SubmitBlocks" does a "for loop Enqueue" calls which is thread-safe
    // but for this code to work, "for loop Enqueue" portion must be a
    // critical section, instead of the internal queue's data.
    // (see the above "TODO" section)
    auto* cbRef = &this->cb;
    tp->SubmitBlocks
    (
        blockCount,
        [
            //name,
            blockCount, oldIssueCount, cbRef,
            callerBlockCount, tpb,
            func = std::forward<Lambda>(func)
        ](uint32_t blockStart, [[maybe_unused]] uint32_t blockEnd)
        {
            // Wait the previous kernel to finish
            // CUDA style queue emulation.
            auto atomicCompletedCounter = std::atomic_ref<uint64_t>((*cbRef)->completedKernelCounter);
            while(atomicCompletedCounter < oldIssueCount)
                atomicCompletedCounter.wait(std::numeric_limits<uint64_t>::max());
            // I dont understand this wait, wait until "curIssueCount == oldIssueCount"
            // but what if we skip a beat and curIssueCount becomes larger than wait value?
            // Further thinking, how this is useful at all?
            // Given a value set I, A is technically all elements of the set but one
            // which is B.
            // "wait_predicate" would be better? Probably hard to implement, i dunno.
            // Anyway if this comment stays it works robustly.
            // TODO: Even if this there is no ABA issue, check the perf
            // and fall back to condition variable / mutex
            // compare as such "wait until (curIssueCount >= oldIssueCount)
            //
            // From this point on it previous kernels should be completed
            //
            //
            std::atomic_thread_fence(std::memory_order_seq_cst);

            assert(blockEnd - blockStart == 1);
            globalKCParams.gridSize = callerBlockCount;
            globalKCParams.blockSize = tpb;

            // Do the block stride loop here,
            // each thread may be responsible for one or more blocks
            for(uint32_t bId = blockStart; bId < callerBlockCount; bId += blockCount)
            {
                globalKCParams.blockId = bId;
                for(uint32_t j = 0; j < tpb; j++)
                {
                    // For kernel call params to work
                    globalKCParams.threadId = j;
                    KernelCallParams kp;
                    func(kp);
                }
            }
            atomicCompletedCounter.fetch_add(1);
            atomicCompletedCounter.notify_all();
            //
            std::atomic_thread_fence(std::memory_order_seq_cst);
        },
        blockCount
    );
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueWorkKernel(std::string_view name,
                                  DeviceWorkIssueParams p,
                                  //
                                  Args&&... fArgs) const
{
    IssueKernelInternal<Kernel>
    (
        name, p.workCount, StaticThreadPerBlock1D(), std::forward<Args>(fArgs)...
    );
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCPU::IssueWorkLambda(std::string_view name,
                                  DeviceWorkIssueParams p,
                                  //
                                  Lambda&& func) const
{
    IssueLambdaInternal
    (
        name, p.workCount, StaticThreadPerBlock1D(), std::forward<Lambda>(func)
    );
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueBlockKernel(std::string_view name,
                                   DeviceBlockIssueParams p,
                                   //
                                   Args&&... fArgs) const
{
    // TODO: Reason about this
    uint32_t workCount = p.blockSize * p.gridSize;
    //
    IssueKernelInternal<Kernel>
    (
        name, workCount, p.blockSize, std::forward<Args>(fArgs)...
    );
}

template<class Lambda, uint32_t Bounds>
MRAY_HOST inline
void GPUQueueCPU::IssueBlockLambda(std::string_view name,
                                   DeviceBlockIssueParams p,
                                   //
                                   Lambda&& func) const
{
    // TODO: Reason about this
    uint32_t workCount = p.blockSize * p.gridSize;
    //
    IssueLambdaInternal
    (
        name, workCount, p.blockSize, std::forward<Lambda>(func)
    );
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueWorkKernel(std::string_view,
                                        DeviceWorkIssueParams,
                                        //
                                        Args&&...) const
{
    // Device issue is scary,
    // This invalidates producer/consumer
    // logic of the threads.
    //
    // For example, if each thread launches a kernel
    // but ThreadPool queue is full
    // these threads will be blocked. And there will not
    // be any consumers left to empty the queue; thus,
    // deadlock will occur.
    //
    // We could've make the thread pool's queue infinite
    // but it will suffer from runaway issues
    // when Host (producer) is faster than the Device (consumer).
    // Which is almost always will be the case.
    //
    // It is not trivial to prevent this,
    // So we disable
    throw MRayError("Device \"CPU\", do not have support for "
                    "in-device kernel launches!");
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueWorkLambda(std::string_view name,
                                        DeviceWorkIssueParams,
                                        //
                                        Lambda&&) const
{
    // See the comment of "DeviceIssueWorkKernel(...)"
    throw MRayError("[{}]: Device \"CPU\", do not have support for "
                    "in-device kernel launches!", name);
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueBlockKernel(std::string_view name,
                                         DeviceBlockIssueParams,
                                         //
                                         Args&&...) const
{
    // See the comment of "DeviceIssueWorkKernel(...)"
    throw MRayError("[{}]: Device \"CPU\", do not have support for "
                    "in-device kernel launches!", name);
}

template<class Lambda, uint32_t Bounds>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueBlockLambda(std::string_view name,
                                         DeviceBlockIssueParams,
                                         //
                                         Lambda&&) const
{
    // See the comment of "DeviceIssueWorkKernel(...)"
    throw MRayError("[{}]: Device \"CPU\", do not have support for "
                    "in-device kernel launches!", name);
}

}


