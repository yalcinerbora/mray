#pragma once

#include "Core/Error.h"
#include "GPUSystemCPU.h"
#include "../GPUSystem.h"   // IWYU pragma: keep

#include "Core/ThreadPool.h"
#include <cstdint>

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

template<uint32_t TPB, class WrappedFunc>
struct GenericWorkFunctor
{
    using ControlBlock = typename GPUQueueCPU::ControlBlockData;

    uint64_t            oldIssueCount;
    uint64_t            newIssueCount;
    ControlBlock*       cbPtr;
    uint32_t            callerBlockCount;
    uint32_t            dynamicTPB;
    WrappedFunc         func;
    AnnotationHandle    domain;
    std::string_view    name;
    mutable uint32_t    localBlockCounter = 0;
    //
    MR_HF_DECL
    void operator()([[maybe_unused]] uint32_t blockStart,
                    [[maybe_unused]] uint32_t blockEnd) const
    {
        // Wait the previous kernel to finish
        // CUDA style queue emulation.
        AtomicWaitValueToAchieve(cbPtr->completedKernelCounter, oldIssueCount);

        // From this point on it previous kernels should be completed
        std::atomic_thread_fence(std::memory_order_seq_cst);
        {
            static const auto thrdWorkAnnot = GPUAnnotationCPU(domain, name);
            const auto thrdWorkScope = thrdWorkAnnot.AnnotateScope();

            assert(blockEnd - blockStart == 1);
            const_cast<uint32_t&>(globalKCParams.gridSize) = callerBlockCount;
            const_cast<uint32_t&>(globalKCParams.blockSize) = TPB;

            // Do the block stride loop here,
            // each thread may be responsible for one or more blocks
            auto blockCounter = std::atomic_ref(localBlockCounter);
            for(uint32_t bId = blockCounter.fetch_add(1u);
                bId < callerBlockCount;
                bId = blockCounter.fetch_add(1u))
            {
                globalKCParams.blockId = bId;
                uint32_t TPB_SELECT = (TPB == 0) ? dynamicTPB : TPB;
                for(uint32_t j = 0; j < TPB_SELECT; j++)
                {
                    globalKCParams.threadId = j;
                    func();
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        {
            static const auto thrdDoneAnnot = GPUAnnotationCPU(domain, "WorkDoneSignal");
            const auto thrdDoneScope = thrdDoneAnnot.AnnotateScope();

            std::atomic_uint64_t& notifyLoc = cbPtr->completedKernelCounter;
            uint64_t oldCount = notifyLoc.fetch_add(1);
            // Only notify when all threads of this block is
            // done to minimize unnecessary wake up.
            if(oldCount + 1 == newIssueCount)
            {
                notifyLoc.notify_all();
            }
        }
    }
};

MR_GF_DEF
KernelCallParamsCPU::KernelCallParamsCPU()
    : gridSize(globalKCParams.gridSize)
    , blockSize(globalKCParams.blockSize)
    , blockId(globalKCParams.blockId)
    , threadId(globalKCParams.threadId)
{}

template<uint32_t TPB, class WrappedKernel>
MR_HF_DEF
void GPUQueueCPU::IssueInternal(std::string_view name,
                                uint32_t totalWorkCount,
                                uint32_t blockSize,
                                WrappedKernel&& kernel) const
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
    uint32_t blockCount = DetermineGridStrideBlock(nullptr, 0, TPB,
                                                   totalWorkCount);
    uint32_t callerBlockCount = Math::DivideUp(totalWorkCount, TPB);
    //
    uint64_t oldIssueCount = cb->issuedKernelCounter.fetch_add(blockCount);
    uint64_t newIssueCount = oldIssueCount + blockCount;

    // Issue kernel
    // "SubmitBlocks" does a "for loop Enqueue" calls which is thread-safe
    // but for this code to work, "for loop Enqueue" portion must be a
    // critical section, instead of the internal queue's data.
    // (see the above "TODO" section)
    tp->SubmitDetachedBlocks
    (
        blockCount,
        GenericWorkFunctor<TPB, WrappedKernel>
        {
            .oldIssueCount = oldIssueCount,
            .newIssueCount = newIssueCount,
            .cbPtr = cb.get(),
            .callerBlockCount = callerBlockCount,
            .dynamicTPB = blockSize,
            .func = std::forward<WrappedKernel>(kernel),
            .domain = this->domain,
            .name = name
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
    assert(p.workCount != 0);
    IssueInternal<StaticThreadPerBlock1D()>
    (
        name, p.workCount, StaticThreadPerBlock1D(),
        [...fArgs = std::forward<Args>(fArgs)]()
        {
            Kernel(fArgs...);
        }
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
    IssueInternal<StaticThreadPerBlock1D()>
    (
        name, p.workCount, StaticThreadPerBlock1D(),
        [func = std::forward<Lambda>(func)]()
        {
            KernelCallParamsCPU kp;
            func(kp);
        }
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
    #define MRAY_KCALL(BLOCK_SIZE)                      \
        IssueInternal<BLOCK_SIZE>                       \
        (                                               \
            name, workCount, BLOCK_SIZE,                \
            [...fArgs = std::forward<Args>(fArgs)]()    \
            {                                           \
                Kernel(fArgs...);                       \
            }                                           \
        )
    // TODO: We are stopping this since it increases compile
    // times. Also compiler does not benefit form it.
    // System is stayed the same but we just call the kernel with
    // zero, and templated function knows what to do.
    // switch (p.blockSize)
    // {
    //     case 1  : MRAY_KCALL(1u);    break;
    //     case 128: MRAY_KCALL(128u);  break;
    //     case 256: MRAY_KCALL(256u);  break;
    //     case 512: MRAY_KCALL(512u);  break;
    //     default : throw MRayError("Unable to Call Kernel {}", name);
    // }
    MRAY_KCALL(0u);

    #undef MRAY_KCALL
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
    #define MRAY_KCALL(BLOCK_SIZE)                  \
        IssueInternal<BLOCK_SIZE>                   \
        (                                           \
            name, workCount, BLOCK_SIZE,            \
            [func = std::forward<Lambda>(func)]()   \
            {                                       \
                KernelCallParamsCPU kp;             \
                func(kp);                           \
            }                                       \
        )
    // TODO: We are stopping this since it increases compile
    // times. Also compiler does not benefit form it.
    // System is stayed the same but we just call the kernel with
    // zero, and templated function knows what to do.
    // switch (p.blockSize)
    // {
    //     case 1  : MRAY_KCALL(1u);    break;
    //     case 128: MRAY_KCALL(128u);  break;
    //     case 256: MRAY_KCALL(256u);  break;
    //     case 512: MRAY_KCALL(512u);  break;
    //     default : throw MRayError("Unable to Call Kernel {}", name);
    // }
    MRAY_KCALL(0u);
    #undef MRAY_KCALL
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


