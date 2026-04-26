#pragma once

#include "GPUSystemHIP.h"
#include "../GPUSystem.h"

static constexpr uint32_t WarpSize()
{
    // Recent AMD GPUs have variable warp size
    // To make the code portable we do not care variable
    // warp size mode, and use highest amount of thread-per-warp (TPW)
    // for each arch.
    //
    //
    //
    // This code must match with the w/e parameter AMD uses when launching /
    // compiling the kernels.
    //
    // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    //
    //
    // I was going to do a big ifdef block here, but after checking the
    // table above, we can get away with 64 for all supported archs.
    //
    // According to this HIP runtime does not support it?
    // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
    //
    // So for RDNA 1-2-3, it is 32, and for the rest it is 64
    //
    // All of our code should work, however; we may not get the best perf
    // for each arch (fine-tune)
    #ifdef __HIP_DEVICE_COMPILE__
        return 64;
    // This may creep in .cpp files, so we return zero and hope for crash.
    #else
        return 0;
    #endif
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MR_GF_DECL
void WarpSynchronize()
{
    // https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
    // After reading this to understand the generic architecture,
    // AMD has SIMD units and custom scheduler to juggle these units.
    // So we do not have sub warp and this can be a noop?
    //
    // (TODO: Check this after we at least compile and check the algorithms
    // that uses sub-warps)
    //
    // So we just static assert if LOGICAL_WARP_SIZE is used besides WarpSize().
    static_assert(LOGICAL_WARP_SIZE == WarpSize(),
                  "On AMD, we can't sync warps at all at the moment and sub-warp"
                  "algorithms should not be used");
    // Noop
}

MR_GF_DECL
static void BlockSynchronize()
{
    // Dirty fix to make host side happy
    #ifdef __HIP_DEVICE_COMPILE__
    __syncthreads();
    #endif
}

MR_GF_DECL
static void ThreadFenceGrid()
{
    // Dirty fix to make host side happy
    #ifdef __HIP_DEVICE_COMPILE__
    __threadfence();
    #endif
}

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


