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
MRAY_HOST inline
void GPUQueueCUDA::IssueKernel(std::string_view name,
                               KernelIssueParams p,
                               Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    using namespace CudaKernelCalls;
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCUDA::IssueLambda(std::string_view name,
                               KernelIssueParams p,
                               //
                               Lambda&& func) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingKernel(std::string_view name,
                                         KernelIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    using namespace CudaKernelCalls;

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
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingLambda(std::string_view name,
                                         KernelIssueParams p,
                                         //
                                         Lambda&& func) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    const void* kernelPtr = reinterpret_cast<const void*>(&KernelCallLambdaCUDA<Lambda, StaticThreadPerBlock1D()>);
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
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernel(std::string_view name,
                                    KernelExactIssueParams p,
                                    //
                                    Args&&... fArgs) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.gridSize != 0);
    using namespace CudaKernelCalls;
    Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactLambda(std::string_view name,
                                    KernelExactIssueParams p,
                                    //
                                    Lambda&& func) const
{
    static const auto annotation = GPUAnnotationCUDA(nvtxDomain, name);
    const auto _ = annotation.AnnotateScope();

    assert(p.gridSize != 0);
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

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueKernel(std::string_view name,
                                     KernelIssueParams p,
                                     Args&&... fArgs) const
{
    assert(p.workCount != 0);
    using namespace CudaKernelCalls;
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueLambda(std::string_view name,
                                     KernelIssueParams p,
                                     //
                                     Lambda&& func) const
{
    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    uint32_t blockCount = Math::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueSaturatingKernel(std::string_view name,
                                               KernelIssueParams p,
                                               //
                                               Args&&... fArgs) const
{
    assert(p.workCount != 0);
    using namespace CudaKernelCalls;

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
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueSaturatingLambda(std::string_view name,
                                               KernelIssueParams p,
                                               //
                                               Lambda&& func) const
{
    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    const void* kernelPtr = reinterpret_cast<const void*>(&KernelCallLambdaCUDA<Lambda, StaticThreadPerBlock1D()>);
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
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueExactKernel(std::string_view name,
                                          KernelExactIssueParams p,
                                          //
                                          Args&&... fArgs) const
{
    assert(p.gridSize != 0);
    using namespace CudaKernelCalls;
    Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_GPU inline
void GPUQueueCUDA::DeviceIssueExactLambda(std::string_view name,
                                          KernelExactIssueParams p,
                                          //
                                          Lambda&& func) const
{
    assert(p.gridSize != 0);
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


