#pragma once

#include "GPUSystemCUDA.h"
#include "GPUSystem.h"

namespace CudaKernelCalls
{
    using namespace mray::cuda;

    template <class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(Bounds)
    static void KernelCallLambdaCUDA(Lambda func)
    {
        func(KernelCallParamsCUDA());
    }
}

namespace mray::cuda
{
MRAY_GPU MRAY_CGPU_INLINE
KernelCallParamsCUDA::KernelCallParamsCUDA()
    : gridSize(gridDim.x)
    , blockSize(blockDim.x)
    , blockId(blockIdx.x)
    , threadId(threadIdx.x)
{}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernel(KernelIssueParams p,
                               Args&&... fArgs) const
{
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueLambda(KernelIssueParams p,
                                //
                                Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingKernel(KernelIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    using namespace CudaKernelCalls;

    const void* kernelPtr = static_cast<const void*>(Kernel);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);

    Kernel<<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingLambda(KernelIssueParams p,
                                         //
                                         Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    const void* kernelPtr = static_cast<const void*>(&KernelCallLambdaCUDA<Lambda, StaticThreadPerBlock1D()>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);


    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueExactKernel(KernelExactIssueParams p,
                                    //
                                    Args&&... fArgs) const
{
    using namespace CudaKernelCalls;
    Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactLambda(KernelExactIssueParams p,
                                    //
                                    Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;

    KernelCallLambdaCUDA<Lambda, Bounds>
    <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}


}


