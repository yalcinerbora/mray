#pragma once

#include "GPUSystemCPU.h"
#include "../GPUSystem.h"

#include "Core/ThreadPool.h"

static constexpr uint32_t WarpSize()
{
    return 1u;
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MR_GF_DEF static void WarpSynchronize()
{
    static_assert(LOGICAL_WARP_SIZE == std::numeric_limits<uint32_t>::max(),
                  "CPU Device does not have notion of warps. "
                  "Please write a CPU specific algorithm.");
}

MR_GF_DEF static void BlockSynchronize() {}

MR_GF_DEF static void ThreadFenceGrid()
{
    // TODO: Reason about this
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

namespace mray::host
{

inline
void NotifyWhenValueReaches(GPUQueueCPU::ControlBlockData* cb, uint64_t valueToNotify)
{
    std::atomic_uint64_t& notifyLoc = cb->completedKernelCounter;
    uint64_t oldCount = notifyLoc.fetch_add(1);
    // Only notify when all threads of this block is
    // done to minimize unnecessary wake up.
    if(oldCount + 1 == valueToNotify)
    {
        cb->curBlockCounter = 0;
        notifyLoc.notify_all();
    }
}

MR_GF_DEF
KernelCallParamsCPU::KernelCallParamsCPU()
    : gridSize(globalKCParams.gridSize)
    , blockSize(globalKCParams.blockSize)
    , blockId(globalKCParams.blockId)
    , threadId(globalKCParams.threadId)
{}

template<auto Kernel, class... Args>
MR_HF_DEF
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

    // Give just enough grid to eliminate grid-stride / block-stride
    // loops (for CPU these do more harm than good, unlike GPU)

    uint32_t blockCount = DetermineGridStrideBlock(nullptr, 0, tpb,
                                                   totalWorkCount);
    uint32_t callerBlockCount = Math::DivideUp(totalWorkCount, tpb);
    //
    uint64_t oldIssueCount = cb->issuedKernelCounter.fetch_add(blockCount);
    uint64_t newIssueCount = oldIssueCount + blockCount;

    // Issue kernel
    // "SubmitBlocks" does a "for loop Enqueue" calls which is thread-safe
    // but for this code to work, "for loop Enqueue" portion must be a
    // critical section, instead of the internal queue's data.
    // (see the above "TODO" section)
    //auto* cbRef = &this->cb;
    ControlBlockData* cbPtr = cb.get();
    [[maybe_unused]]
    auto r = tp->SubmitBlocks
    (
        blockCount,
        [
            ... fArgs = std::forward<Args>(fArgs),
            oldIssueCount, newIssueCount,
            cbPtr, callerBlockCount, tpb,
            name, domain = this->domain
            //, totalWorkCount, blockCount
        ]
        (
            [[maybe_unused]] uint32_t blockStart,
            [[maybe_unused]] uint32_t blockEnd
        )
        {
            // Wait the previous kernel to finish
            // CUDA style queue emulation.
            AtomicWaitValueToAchieve(cbPtr->completedKernelCounter, oldIssueCount);

            std::atomic_thread_fence(std::memory_order_seq_cst);
            static const auto threadAnnotation = GPUAnnotationCPU(domain, name);
            const auto threadAnnotationScope = threadAnnotation.AnnotateScope();

            // From this point on it previous kernels should be completed
            assert(blockEnd - blockStart == 1);
            globalKCParams.gridSize = callerBlockCount;
            globalKCParams.blockSize = tpb;

            // Do the block stride loop here,
            // each thread may be responsible for one or more blocks
            //for(uint32_t bId = blockStart; bId < callerBlockCount; bId += blockCount)
            auto& blockCounter = cbPtr->curBlockCounter;
            for(uint32_t bId = blockCounter.fetch_add(1u);
                bId < callerBlockCount;
                bId = blockCounter.fetch_add(1u))
            {
                globalKCParams.blockId = bId;
                for(uint32_t j = 0; j < tpb; j++)
                {
                    // For kernel call params to work
                    globalKCParams.threadId = j;
                    Kernel(fArgs...);
                }
            }
            //
            std::atomic_thread_fence(std::memory_order_seq_cst);
            NotifyWhenValueReaches(cbPtr, newIssueCount);
        },
        blockCount
    );

    if constexpr(MRAY_IS_DEBUG) r.GetAll();
}

template<class Lambda>
MR_HF_DEF
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

    // Give just enough grid to eliminate grid-stride / block-stride
    // loops (for CPU these do more harm than good, unlike GPU)

    uint32_t blockCount = DetermineGridStrideBlock(nullptr, 0, tpb,
                                                   totalWorkCount);
    uint32_t callerBlockCount = Math::DivideUp(totalWorkCount, tpb);
    //
    uint64_t oldIssueCount = cb->issuedKernelCounter.fetch_add(blockCount);
    uint64_t newIssueCount = oldIssueCount + blockCount;

    // Issue kernel
    // "SubmitBlocks" does a "for loop Enqueue" calls which is thread-safe
    // but for this code to work, "for loop Enqueue" portion must be a
    // critical section, instead of the internal queue's data.
    // (see the above "TODO" section)
    //auto* cbRef = &this->cb;
    ControlBlockData* cbPtr = cb.get();
    [[maybe_unused]]
    auto r = tp->SubmitBlocks
    (
        blockCount,
        [
            oldIssueCount, newIssueCount,
            cbPtr, callerBlockCount, tpb,
            func = std::forward<Lambda>(func),
            name, domain = this->domain
            //, totalWorkCount, blockCount
        ]
        (
            [[maybe_unused]] uint32_t blockStart,
            [[maybe_unused]] uint32_t blockEnd
        )
        {
            // Wait the previous kernel to finish
            // CUDA style queue emulation.
            AtomicWaitValueToAchieve(cbPtr->completedKernelCounter, oldIssueCount);

            // From this point on it previous kernels should be completed
            std::atomic_thread_fence(std::memory_order_seq_cst);
            static const auto threadAnnotation = GPUAnnotationCPU(domain, name);
            const auto threadAnnotationScope = threadAnnotation.AnnotateScope();

            assert(blockEnd - blockStart == 1);
            globalKCParams.gridSize = callerBlockCount;
            globalKCParams.blockSize = tpb;

            // Do the block stride loop here,
            // each thread may be responsible for one or more blocks
            //for(uint32_t bId = blockStart; bId < callerBlockCount; bId += blockCount)
            auto& blockCounter = cbPtr->curBlockCounter;
            for(uint32_t bId = blockCounter.fetch_add(1u);
                bId < callerBlockCount;
                bId = blockCounter.fetch_add(1u))
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
            //
            std::atomic_thread_fence(std::memory_order_seq_cst);
            NotifyWhenValueReaches(cbPtr, newIssueCount);
        },
        blockCount
    );

    if constexpr(MRAY_IS_DEBUG) r.GetAll();
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueWorkKernel(std::string_view name,
                                  DeviceWorkIssueParams p,
                                  //
                                  Args&&... fArgs) const
{
    assert(p.workCount != 0);
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
    assert(p.workCount != 0);
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
    assert(p.gridSize != 0);
    assert(p.blockSize != 0);
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
    assert(p.gridSize != 0);
    assert(p.blockSize != 0);

    // TODO: Reason about this
    uint32_t workCount = p.blockSize * p.gridSize;
    //
    IssueLambdaInternal
    (
        name, workCount, p.blockSize, std::forward<Lambda>(func)
    );
}

template<auto Kernel, class... Args>
MR_GF_DEF
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
MR_GF_DEF
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
MR_GF_DEF
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
MR_GF_DEF
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


