#pragma once

#include "GPUSystemHIP.h"
#include "../GPUSystem.h"

#include <rocprim/rocprim.hpp>

static constexpr uint32_t WarpSize()
{
    return rocprim::device_warp_size();
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MRAY_GPU MRAY_GPU_INLINE
static void WarpSynchronize()
{
    // Dirty fix to make host side happy
    #ifdef __HIP_DEVICE_COMPILE__
    static_assert(LOGICAL_WARP_SIZE == 1 || LOGICAL_WARP_SIZE == 2 ||
                  LOGICAL_WARP_SIZE == 4 || LOGICAL_WARP_SIZE == 8 ||
                  LOGICAL_WARP_SIZE == 16 || LOGICAL_WARP_SIZE == 32,
                  "Logical warps must be power of 2 and \"<32\"");

    // Technically single-threaded logical warp is self synchronizing,
    // so no need to do sync.
    if constexpr(LOGICAL_WARP_SIZE != 1)
    {
        static constexpr uint32_t FULL_MASK = std::numeric_limits<uint32_t>::max();
        // Creating all FF's is UB (when doint it via shift, is there any other way to do it)
        // since we shift out of bounds so...
        static constexpr uint32_t MASK = (LOGICAL_WARP_SIZE == 32)
            ? FULL_MASK
            : (1u << LOGICAL_WARP_SIZE) - 1u;
        uint32_t localWarpId = threadIdx.x % WarpSize();
        uint32_t logicalWarpId = localWarpId / LOGICAL_WARP_SIZE;
        uint32_t localMask = MASK << (logicalWarpId * LOGICAL_WARP_SIZE);
        // TODO: no "__syncwarp(...)" in AMD, should we need it?
        // __syncwarp(localMask);
    }
    #endif
}

MRAY_GPU MRAY_GPU_INLINE
static void BlockSynchronize()
{
    // Dirty fix to make host side happy
    #ifdef __HIP_DEVICE_COMPILE__
    __syncthreads();
    #endif
}

MRAY_GPU MRAY_GPU_INLINE
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

MRAY_GPU MRAY_CGPU_INLINE
KernelCallParamsHIP::KernelCallParamsHIP()
    : gridSize(gridDim.x)
    , blockSize(blockDim.x)
    , blockId(blockIdx.x)
    , threadId(threadIdx.x)
{}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueHIP::IssueKernel(std::string_view name,
                              KernelIssueParams p,
                              Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationHIP(roctxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    using namespace HipKernelCalls;
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    HIP_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueHIP::IssueLambda(std::string_view name,
                              KernelIssueParams p,
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
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallLambdaHIP<Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    HIP_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueHIP::IssueSaturatingKernel(std::string_view name,
                                        KernelIssueParams p,
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
void GPUQueueHIP::IssueSaturatingLambda(std::string_view name,
                                        KernelIssueParams p,
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
void GPUQueueHIP::IssueExactKernel(std::string_view name,
                                   KernelExactIssueParams p,
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
void GPUQueueHIP::IssueExactLambda(std::string_view name,
                                   KernelExactIssueParams p,
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
void GPUQueueHIP::DeviceIssueKernel(std::string_view name,
                                    KernelIssueParams p,
                                    Args&&... fArgs) const
{
    // assert(p.workCount != 0);
    // using namespace HipKernelCalls;
    // uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    // uint32_t blockSize = StaticThreadPerBlock1D();

    // Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Args>(fArgs)...
    // );
    // HIP_KERNEL_CHECK();
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueLambda(std::string_view name,
                                    KernelIssueParams p,
                                    //
                                    Lambda&& func) const
{
    // assert(p.workCount != 0);
    // static_assert(std::is_rvalue_reference_v<decltype(func)>,
    //               "Not passing Lambda as rvalue_reference. This kernel call "
    //               "would've been failed in runtime!");
    // using namespace HipKernelCalls;
    // uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    // uint32_t blockSize = StaticThreadPerBlock1D();

    // KernelCallLambdaHIP<Lambda>
    // <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Lambda>(func)
    // );
    // HIP_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueSaturatingKernel(std::string_view name,
                                              KernelIssueParams p,
                                              //
                                              Args&&... fArgs) const
{
    // assert(p.workCount != 0);
    // using namespace HipKernelCalls;

    // const void* kernelPtr = reinterpret_cast<const void*>(Kernel);
    // uint32_t threadCount = StaticThreadPerBlock1D();
    // uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
    //                                                p.sharedMemSize,
    //                                                threadCount,
    //                                                p.workCount);

    // Kernel<<<blockCount, threadCount, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Args>(fArgs)...
    // );
    // HIP_KERNEL_CHECK();
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueSaturatingLambda(std::string_view name,
                                              KernelIssueParams p,
                                              //
                                              Lambda&& func) const
{
    // assert(p.workCount != 0);
    // static_assert(std::is_rvalue_reference_v<decltype(func)>,
    //               "Not passing Lambda as rvalue_reference. This kernel call "
    //               "would've been failed in runtime!");
    // using namespace HipKernelCalls;
    // const void* kernelPtr = reinterpret_cast<const void*>(&KernelCallLambdaHIP<Lambda, StaticThreadPerBlock1D()>);
    // uint32_t threadCount = StaticThreadPerBlock1D();
    // uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
    //                                                p.sharedMemSize,
    //                                                threadCount,
    //                                                p.workCount);


    // KernelCallLambdaHIP<Lambda>
    // <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Lambda>(func)
    // );
    // HIP_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueExactKernel(std::string_view name,
                                         KernelExactIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    // assert(p.gridSize != 0);
    // using namespace HipKernelCalls;
    // Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Args>(fArgs)...
    // );
    // HIP_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_GPU inline
void GPUQueueHIP::DeviceIssueExactLambda(std::string_view name,
                                         KernelExactIssueParams p,
                                         //
                                         Lambda&& func) const
{
    // assert(p.gridSize != 0);
    // static_assert(std::is_rvalue_reference_v<decltype(func)>,
    //               "Not passing Lambda as rvalue_reference. This kernel call "
    //               "would've been failed in runtime!");
    // using namespace HipKernelCalls;

    // KernelCallLambdaHIP<Lambda, Bounds>
    // <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    // (
    //     std::forward<Lambda>(func)
    // );
    // HIP_KERNEL_CHECK();
}

}


