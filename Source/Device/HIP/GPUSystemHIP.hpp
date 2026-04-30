#pragma once

#include "GPUSystemHIP.h"
#include "../GPUSystem.h"

namespace HipKernelCalls
{
    using namespace mray::hip;

    template <class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(Bounds)
    static void KernelCallLambdaHIP(Lambda func)
    {
        func(KernelCallParamsHIP());
    }
}

namespace mray::hip
{

MR_GF_DEF
KernelCallParamsHIP::KernelCallParamsHIP()
    : gridSize(gridDim.x)
    , blockSize(blockDim.x)
    , blockId(blockIdx.x)
    , threadId(threadIdx.x)
{}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueHIP::IssueWorkKernel(std::string_view name,
                                  DeviceWorkIssueParams p,
                                  //
                                  Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationHIP(roctxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    using namespace HipKernelCalls;

    const void* kernelPtr = reinterpret_cast<const void*>(Kernel);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);

    Kernel<<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    HIP_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueHIP::IssueWorkLambda(std::string_view name,
                                  DeviceWorkIssueParams p,
                                  //
                                  Lambda&& func) const
{
    static const auto annotation = GPUAnnotationHIP(roctxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace HipKernelCalls;
    const void* kernelPtr = reinterpret_cast<const void*>(&KernelCallLambdaHIP<Lambda, StaticThreadPerBlock1D()>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);


    KernelCallLambdaHIP<Lambda>
    <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    HIP_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueHIP::IssueBlockKernel(std::string_view name,
                                   DeviceBlockIssueParams p,
                                   //
                                   Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationHIP(roctxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.gridSize != 0);
    using namespace HipKernelCalls;
    Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    HIP_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_HOST inline
void GPUQueueHIP::IssueBlockLambda(std::string_view name,
                                   DeviceBlockIssueParams p,
                                   //
                                   Lambda&& func) const
{
    static const auto annotation = GPUAnnotationHIP(roctxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.gridSize != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace HipKernelCalls;

    KernelCallLambdaHIP<Lambda, Bounds>
    <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    HIP_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueWorkKernel(std::string_view name,
                                        DeviceWorkIssueParams p,
                                        //
                                        Args&&... fArgs) const
{
    // TODO: Properly handle these
    assert(false && "Not yet Implemented (HIP does not support it maybe?)");
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueWorkLambda(std::string_view name,
                                        DeviceWorkIssueParams p,
                                        //
                                        Lambda&& func) const
{
    // TODO: Properly handle these
    assert(false && "Not yet Implemented (HIP does not support it maybe?)");
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueBlockKernel(std::string_view name,
                                         DeviceBlockIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    // TODO: Properly handle these
    assert(false && "Not yet Implemented (HIP does not support it maybe?)");
}

template<class Lambda, uint32_t Bounds>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueBlockLambda(std::string_view name,
                                         DeviceBlockIssueParams p,
                                         //
                                         Lambda&& func) const
{
    // TODO: Properly handle these
    assert(false && "Not yet Implemented (HIP does not support it maybe?)");
}

}


