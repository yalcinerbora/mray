#pragma once

#include "GPUSystemCUDA.h"
#include "../GPUSystem.h"

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
void GPUQueueCUDA::IssueWorkKernel(std::string_view name,
                                   DeviceWorkIssueParams p,
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
void GPUQueueCUDA::IssueWorkLambda(std::string_view name,
                                   DeviceWorkIssueParams p,
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
void GPUQueueCUDA::IssueBlockKernel(std::string_view name,
                                    DeviceBlockIssueParams p,
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
void GPUQueueCUDA::IssueBlockLambda(std::string_view name,
                                    DeviceBlockIssueParams p,
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
void GPUQueueCUDA::DeviceIssueWorkKernel(std::string_view name,
                                               DeviceWorkIssueParams p,
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
void GPUQueueCUDA::DeviceIssueWorkLambda(std::string_view name,
                                               DeviceWorkIssueParams p,
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
void GPUQueueCUDA::DeviceIssueBlockKernel(std::string_view name,
                                            DeviceBlockIssueParams p,
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
void GPUQueueCUDA::DeviceIssueBlockLambda(std::string_view name,
                                          DeviceBlockIssueParams p,
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


