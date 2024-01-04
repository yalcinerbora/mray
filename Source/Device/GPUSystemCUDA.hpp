
#pragma once

#include "GPUSystemCUDA.h"
#include "GPUSystem.h"

namespace CudaKernelCalls
{
    using namespace mray::cuda;

    // Passing the device function directly as a template parameter
    // So that NVCC distinguishes host/device'ness of the function pt
    template <auto DeviceFunction, class... Args>
    MRAY_KERNEL
    static void KernelCallCUDA(Args... fArgs)
    {
        // Grid-stride loop
        KernelCallParams launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        DeviceFunction(launchParams, std::forward<Args>(fArgs)...);
    }

    template <uint32_t TPB, auto DeviceFunction, class... Args>
    __launch_bounds__(TPB)
    MRAY_KERNEL
    static void KernelCallBoundedCUDA(Args... fArgs)
    {
        // Grid-stride loop
        KernelCallParams launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        DeviceFunction(launchParams, std::forward<Args>(fArgs)...);
    }


    template <class Lambda>
    MRAY_KERNEL
    static void KernelCallLambdaCUDA(Lambda func)
    {
        // Grid-stride loop
        KernelCallParams launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        func(launchParams);
    }

    template <uint32_t TPB, class Lambda>
    __launch_bounds__(TPB)
    MRAY_KERNEL
    static void KernelCallBoundedLambdaCUDA(Lambda func)
    {
        // Grid-stride loop
        KernelCallParams launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        func(launchParams);
    }
}

namespace mray::cuda
{

template<auto DeviceFunction, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernel(KernelIssueParams p,
                               Args&&... fArgs) const
{
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallBoundedCUDA<StaticThreadPerBlock1D(), DeviceFunction, Args...>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        fArgs...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernelL(KernelIssueParams p,
                                //
                                Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernell call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallBoundedLambdaCUDA<StaticThreadPerBlock1D(), Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto DeviceFunction, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingKernel(KernelIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    using namespace CudaKernelCalls;

    const void* kernelPtr = static_cast<const void*>(&KernelCallBoundedCUDA<StaticThreadPerBlock1D(), DeviceFunction, Args...>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);


    KernelCallBoundedCUDA<StaticThreadPerBlock1D(), DeviceFunction, Args...>
    <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        fArgs...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingKernelL(KernelIssueParams p,
                                          //
                                          Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernell call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    const void* kernelPtr = static_cast<const void*>(&KernelCallBoundedLambdaCUDA<StaticThreadPerBlock1D(), Lambda>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);


    KernelCallBoundedLambdaCUDA<StaticThreadPerBlock1D(), Lambda>
    <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        func
    );
    CUDA_KERNEL_CHECK();
}

template<auto DeviceFunction, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueExactKernel(KernelExactIssueParams p,
                                    //
                                    Args&&... fArgs) const
{
    using namespace CudaKernelCalls;
    KernelCallCUDA<DeviceFunction, Args...>
    <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        fArgs...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernelL(KernelExactIssueParams p,
                                    //
                                    Lambda&& func) const
{
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernell call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;

    KernelCallLambdaCUDA<Lambda>
    <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        func
    );
    CUDA_KERNEL_CHECK();
}


}


